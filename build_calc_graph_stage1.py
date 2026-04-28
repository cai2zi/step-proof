"""
阶段一：从 parquet 目录批量建图，写入 graph-v1 JSONL。

核心特性：
  - 本地 vLLM (vllm.LLM)：进程内加载模型，零 HTTP 开销
  - Sliding-pool batch：成功写入、失败留池、token overflow 跳过，GPU 始终满负载
  - Resume：读取已有输出 JSONL，跳过已完成 record_id
  - Orphan-drop 兜底：超出 max_retries 后尝试清理孤立节点再接受
  - 独立输出：graphs.jsonl / skipped.jsonl / failed.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from proofflow.fdg_graph import (
    FDGDocument,
    build_fdg_messages,
    fdg_final_fact_ids,
    fdg_topo_order,
    parse_and_validate_fdg,
)
from proofflow.graph_mode import FDG_GRAPH_MODE, LEGACY_GRAPH_MODE
from proofflow.node_schema import infer_role, is_final, is_structural_final
from proofflow.local_vllm import LocalLLMManager
from proofflow.proof_graph import (
    ProofGraphItem,
    append_error_to_messages,
    build_graph_messages,
    parse_and_validate_graph,
    try_orphan_drop_recovery,
)

load_dotenv()

DEFAULT_MODEL_PATH = os.getenv("GRAPH_MODEL_PATH", "/data/czx/models/Qwen3.5-9B")
DEFAULT_TP = int(os.getenv("GRAPH_TP", "4"))
DEFAULT_GPUS = os.getenv("GRAPH_GPUS", "0,1,2,3")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendingRecord:
    record_id: str
    problem: str
    raw_cot: str
    source_file: str
    source_row_pos: int
    messages: List[Dict[str, str]]
    retry_count: int = 0
    last_parsed_graph: Optional[List[dict]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers: parquet iteration, serialization, resume
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_columns(df_columns: Any, required: List[str], path: Path) -> None:
    missing = [c for c in required if c not in df_columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; have {list(df_columns)}")


def _iter_parquet_rows(
    parquet_dir: Path,
    glob_pat: str,
    required_cols: List[str],
) -> Iterable[Tuple[Path, int, pd.Series]]:
    for fp in sorted(parquet_dir.glob(glob_pat)):
        if not fp.is_file():
            continue
        df = pd.read_parquet(fp)
        _require_columns(df.columns, required_cols, fp)
        for pos, (_, row) in enumerate(df.iterrows()):
            yield fp, pos, row


def _rel_source_file(parquet_dir: Path, fp: Path) -> str:
    try:
        return str(fp.resolve().relative_to(parquet_dir.resolve()))
    except ValueError:
        return str(fp.resolve())


def _topo_order_from_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
    ids = [n["id"] for n in nodes]
    id_set = set(ids)
    succ: Dict[str, List[str]] = {i: [] for i in id_set}
    indeg: Dict[str, int] = {i: 0 for i in id_set}
    for n in nodes:
        for d in (n.get("dependencies") or []):
            if d in id_set:
                indeg[n["id"]] += 1
                succ[d].append(n["id"])

    q = deque(sorted(i for i in id_set if indeg[i] == 0))
    order: List[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == len(id_set) else list(ids)


def _serialize_graph_nodes(
    proof_items: List[ProofGraphItem],
    id_schema_mode: str,
) -> List[Dict[str, Any]]:
    out = []
    for item in proof_items:
        nt = getattr(item, "node_type", None)
        nv = getattr(item, "needs_verification", None)
        out.append({
            "id": item.id,
            "role": infer_role(item.id, nt, mode=id_schema_mode),
            "node_type": nt or "",
            "natural_language": item.natural_language,
            "statement": item.statement,
            "dependencies": list(item.dependencies or []),
            "needs_verification": nv,
        })
    return out


def _final_node_ids(nodes: List[Dict[str, Any]], id_schema_mode: str) -> List[str]:
    return sorted(
        n["id"] for n in nodes
        if is_structural_final(n["id"], id_schema_mode)
        or is_final(n["id"], n.get("node_type") or None, id_schema_mode)
    )


def load_done_ids(out_path: Path) -> set:
    """Read already-completed record_ids from an existing output JSONL."""
    done: set = set()
    if not out_path.is_file():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("meta", {}).get("record_id")
                if rid:
                    done.add(str(rid))
            except json.JSONDecodeError:
                pass
    return done


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _build_legacy_payload(
    record: PendingRecord,
    items: List[ProofGraphItem],
    id_schema_mode: str,
    tries: int,
    include_think_in_dag: bool,
    extraction_response: Optional[str],
) -> Dict[str, Any]:
    nodes = _serialize_graph_nodes(items, id_schema_mode)
    return {
        "meta": {
            "schema_version": "graph-v1",
            "graph_mode": LEGACY_GRAPH_MODE,
            "record_id": record.record_id,
            "task_profile": "calc",
            "source_file": record.source_file,
            "source_row_pos": record.source_row_pos,
            "created_at": _utc_now_iso(),
            "graph_build_tries": tries,
            "id_schema_mode": id_schema_mode,
            "include_think_in_dag": include_think_in_dag,
        },
        "input": {
            "problem": record.problem,
            "raw_cot": record.raw_cot,
        },
        "extraction": {
            "raw_response": extraction_response,
        },
        "graph": {
            "nodes": nodes,
            "topo_order": _topo_order_from_nodes(nodes),
            "final_nodes": _final_node_ids(nodes, id_schema_mode),
        },
    }


def _build_fdg_payload(
    record: PendingRecord,
    document: FDGDocument,
    tries: int,
    include_think_in_dag: bool,
    extraction_response: Optional[str],
    validation_warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "schema_version": "fdg-v1",
            "graph_mode": FDG_GRAPH_MODE,
            "record_id": record.record_id,
            "task_profile": "calc",
            "source_file": record.source_file,
            "source_row_pos": record.source_row_pos,
            "created_at": _utc_now_iso(),
            "graph_build_tries": tries,
            "include_think_in_dag": include_think_in_dag,
        },
        "input": {
            "problem": record.problem,
            "raw_cot": record.raw_cot,
        },
        "extraction": {
            "raw_response": extraction_response,
        },
        "graph": {
            "facts": [fact.model_dump() for fact in document.facts],
            "topo_order": fdg_topo_order(document.facts),
            "final_fact_ids": fdg_final_fact_ids(document.facts),
            "validation_warnings": validation_warnings,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (batch + local vLLM): build calc proof graphs → graph-v1 JSONL.",
    )
    # Input
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="Directory containing .parquet shards")
    parser.add_argument("--glob", default="*.parquet",
                        help="Glob pattern under parquet-dir (default: *.parquet)")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max NEW records to process (-1 = all)")
    # Output
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "graphs.jsonl",
                        help="Successful graphs output JSONL")
    parser.add_argument("--skipped", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "skipped.jsonl",
                        help="Token-overflow skipped records JSONL")
    parser.add_argument("--failed", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "failed.jsonl",
                        help="Permanently failed records JSONL")
    # Resume
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing output and start fresh (overwrite)")
    # vLLM
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP)
    parser.add_argument("--gpus", default=DEFAULT_GPUS,
                        help="CUDA_VISIBLE_DEVICES for the local vLLM engine")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help='JSON object for tokenizer.apply_chat_template (default: {"enable_thinking": false})',
    )
    parser.add_argument("--token-limit", type=int, default=40960,
                        help="Max prompt tokens; longer prompts are skipped")
    # Batch / retry
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    # Graph options
    parser.add_argument(
        "--graph-mode",
        choices=(LEGACY_GRAPH_MODE, FDG_GRAPH_MODE),
        default=LEGACY_GRAPH_MODE,
    )
    parser.add_argument("--id-schema-mode", default="calc")
    parser.add_argument("--validation-profile", default="strict")
    parser.add_argument("--follow-dag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-graph-rewrite-after", type=int, default=3)
    parser.add_argument(
        "--include-think-in-dag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow DAG extraction to see <think>...</think> content in the response. "
            "Use --no-include-think-in-dag to hide think blocks from the graph model; "
            "the full raw_cot is still written to output JSONL."
        ),
    )

    args = parser.parse_args()

    chat_template_kwargs: Optional[Dict[str, Any]] = None
    if args.chat_template_kwargs_json:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit("--chat-template-kwargs-json must be a JSON object")

    if not args.parquet_dir.is_dir():
        raise SystemExit(f"--parquet-dir is not a directory: {args.parquet_dir}")

    # Set CUDA_VISIBLE_DEVICES before importing vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Resume: load already-done IDs
    done_ids: set = set()
    if not args.no_resume:
        done_ids = load_done_ids(args.out)
        if done_ids:
            print(f"[resume] skipping {len(done_ids)} already-processed record(s)")

    # Prepare output dirs
    for p in (args.out, args.skipped, args.failed):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Load local vLLM
    print(f"[init] loading model {args.model_path} (tp={args.tensor_parallel_size}, gpus={args.gpus}) ...")
    llm = LocalLLMManager(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
        top_k=args.top_k,
        token_limit=args.token_limit,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        chat_template_kwargs=chat_template_kwargs,
    )
    print("[init] model ready.\n")

    # Parquet iterator with column check
    required = [args.id_column, args.question_column, args.response_column]
    raw_iter = _iter_parquet_rows(args.parquet_dir, args.glob, required)

    new_count = 0  # newly pulled records (for --limit)

    def next_pending() -> Optional[PendingRecord]:
        """Pull the next not-yet-done record from parquet."""
        nonlocal new_count
        while True:
            if args.limit >= 0 and new_count >= args.limit:
                return None
            item = next(raw_iter, None)
            if item is None:
                return None
            fp, pos, row = item
            record_id = str(row[args.id_column]).strip()
            if not record_id:
                continue
            if record_id in done_ids:
                print(f"  [resume] skip {record_id}")
                continue
            problem = row[args.question_column]
            raw_cot = row[args.response_column]
            if problem is None or (isinstance(problem, float) and pd.isna(problem)):
                continue
            if raw_cot is None or (isinstance(raw_cot, float) and pd.isna(raw_cot)):
                continue
            if args.graph_mode == FDG_GRAPH_MODE:
                msgs = build_fdg_messages(
                    problem_text=str(problem),
                    solution_or_cot=str(raw_cot),
                    include_think_in_dag=args.include_think_in_dag,
                )
            else:
                msgs = build_graph_messages(
                    task_profile="calc",
                    problem=str(problem),
                    raw_cot=str(raw_cot),
                    include_think_in_dag=args.include_think_in_dag,
                )
            new_count += 1
            return PendingRecord(
                record_id=record_id,
                problem=str(problem),
                raw_cot=str(raw_cot),
                source_file=_rel_source_file(args.parquet_dir, fp),
                source_row_pos=pos,
                messages=msgs,
            )

    # Pre-fill pool
    pool: deque[PendingRecord] = deque()
    while len(pool) < args.batch_size:
        rec = next_pending()
        if rec is None:
            break
        pool.append(rec)

    stats = {"ok": 0, "skipped": 0, "failed": 0, "retried": 0}

    write_mode = "w" if args.no_resume else "a"
    with (
        open(args.out, write_mode, encoding="utf-8") as graphs_f,
        open(args.skipped, write_mode, encoding="utf-8") as skipped_f,
        open(args.failed, write_mode, encoding="utf-8") as failed_f,
    ):
        while pool:
            batch = list(pool)
            pool.clear()

            print(f"\n[batch] size={len(batch)}  (ok={stats['ok']} skip={stats['skipped']} fail={stats['failed']})")
            contents = llm.batch_generate([r.messages for r in batch])

            for record, content in zip(batch, contents):

                # ── token overflow ──────────────────────────────────────────
                if content is None:
                    skipped_f.write(json.dumps({
                        "record_id": record.record_id,
                        "source_file": record.source_file,
                        "source_row_pos": record.source_row_pos,
                        "reason": "token_overflow",
                        "input": {
                            "problem": record.problem,
                            "raw_cot": record.raw_cot,
                        },
                        "extraction": {
                            "raw_response": None,
                        },
                    }, ensure_ascii=False) + "\n")
                    skipped_f.flush()
                    stats["skipped"] += 1
                    print(f"  [skip]  {record.record_id} (token overflow)")
                    continue

                # ── parse + validate ────────────────────────────────────────
                if args.graph_mode == FDG_GRAPH_MODE:
                    result = parse_and_validate_fdg(content)
                else:
                    result = parse_and_validate_graph(
                        content=content,
                        id_schema_mode=args.id_schema_mode,
                        validation_profile=args.validation_profile,
                        follow_dag=args.follow_dag,
                        attempt=record.retry_count,
                        allow_graph_rewrite_after=args.allow_graph_rewrite_after,
                    )

                if result.ok:
                    if args.graph_mode == FDG_GRAPH_MODE:
                        payload = _build_fdg_payload(
                            record,
                            result.document,
                            record.retry_count + 1,
                            args.include_think_in_dag,
                            content,
                            list((result.report or {}).get("warnings") or []),
                        )
                    else:
                        payload = _build_legacy_payload(
                            record,
                            result.items,
                            args.id_schema_mode,
                            record.retry_count + 1,
                            args.include_think_in_dag,
                            content,
                        )
                    graphs_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    graphs_f.flush()
                    stats["ok"] += 1
                    print(f"  [ok]    {record.record_id}  (tries={record.retry_count + 1})")
                    continue

                # keep last_parsed_graph for orphan-drop
                if args.graph_mode == LEGACY_GRAPH_MODE and result.last_parsed_graph is not None:
                    record.last_parsed_graph = result.last_parsed_graph

                # ── max retries reached → orphan-drop fallback ──────────────
                if record.retry_count >= args.max_retries:
                    recovered = None
                    if args.graph_mode == LEGACY_GRAPH_MODE and record.last_parsed_graph:
                        recovered = try_orphan_drop_recovery(
                            last_parsed_graph=record.last_parsed_graph,
                            id_schema_mode=args.id_schema_mode,
                            validation_profile=args.validation_profile,
                            follow_dag=args.follow_dag,
                            max_retries=args.max_retries,
                        )
                    if recovered is not None:
                        payload = _build_legacy_payload(
                            record,
                            recovered,
                            args.id_schema_mode,
                            record.retry_count + 1,
                            args.include_think_in_dag,
                            content,
                        )
                        graphs_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        graphs_f.flush()
                        stats["ok"] += 1
                        print(f"  [ok]    {record.record_id}  (orphan-drop recovery)")
                    else:
                        failed_f.write(json.dumps({
                            "record_id": record.record_id,
                            "source_file": record.source_file,
                            "source_row_pos": record.source_row_pos,
                            "retry_count": record.retry_count,
                            "last_error": result.error_msg,
                            "input": {
                                "problem": record.problem,
                                "raw_cot": record.raw_cot,
                            },
                            "extraction": {
                                "raw_response": content,
                            },
                        }, ensure_ascii=False) + "\n")
                        failed_f.flush()
                        stats["failed"] += 1
                        print(f"  [fail]  {record.record_id}  (after {record.retry_count} retries)")
                    continue

                # ── retry: update messages and put back in pool ─────────────
                record.messages = append_error_to_messages(record.messages, result.error_msg)
                record.retry_count += 1
                pool.appendleft(record)   # retries go to the front
                stats["retried"] += 1
                print(f"  [retry] {record.record_id}  (attempt {record.retry_count}/{args.max_retries})")

            # ── refill pool with new records ────────────────────────────────
            while len(pool) < args.batch_size:
                rec = next_pending()
                if rec is None:
                    break
                pool.append(rec)

    print(
        f"\n[done] ok={stats['ok']}  skipped={stats['skipped']}  "
        f"failed={stats['failed']}  retried={stats['retried']}"
    )
    print(f"  graphs  → {args.out}")
    print(f"  skipped → {args.skipped}")
    print(f"  failed  → {args.failed}")


if __name__ == "__main__":
    main()
