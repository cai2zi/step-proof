#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

try:
    from path_defaults import step_proofs_root
except Exception:  # pragma: no cover - keeps the script runnable outside this folder.
    step_proofs_root = None  # type: ignore[assignment]


JsonDict = dict[str, Any]


DEFAULT_API_BASE_URL = "https://poloapi.top/v1"
DEFAULT_API_KEY_ENV = "POLOAI_API_KEY"
DEFAULT_MODEL = "gpt-5-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample rollout records whose nodes formalized successfully but failed proving, "
            "then ask an OpenAI-compatible chat API to explain every failed node in Chinese."
        )
    )
    parser.add_argument(
        "project_root",
        type=Path,
        help=(
            "Project root used to locate outputs. Accepts either D:/program/research "
            "or D:/program/research/step-proof."
        ),
    )
    parser.add_argument(
        "exp_name",
        help="Experiment name, with or without the step_proof_ prefix.",
    )
    parser.add_argument(
        "--step-proof-root",
        type=Path,
        default=None,
        help="Directory containing step_proof_* experiments. If omitted, inferred from project_root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Default: <exp_dir>/analysis/prove_failure_llm_reasons.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of rollout records to analyze when not using --record-id. Default: 3.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random", "stratified-random"],
        default="first",
        help=(
            "Select eligible rollouts. stratified-random samples by per-record prove "
            "verification ratio buckets. Default: first."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --sample-mode=random or stratified-random.",
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=0,
        help=(
            "For --sample-mode=stratified-random, sample this many records from each "
            "prove-ratio bucket. 0 means distribute --limit across buckets. Default: 0."
        ),
    )
    parser.add_argument(
        "--prove-ratio-bucket-step",
        type=int,
        default=20,
        help=(
            "Bucket width in percent for --sample-mode=stratified-random. "
            "Default: 20, producing 0-20%%, ..., 80-100%%, 100%%."
        ),
    )
    parser.add_argument(
        "--include-skipped-derived",
        action="store_true",
        help=(
            "Include derived nodes with skip=1 when computing stratified prove ratios "
            "and eligible failed nodes. Default uses exclude_skipped_derived."
        ),
    )
    parser.add_argument(
        "--record-id",
        default="",
        help="Analyze one exact rollout record id, e.g. aime_2024__70__rollout_1.",
    )
    parser.add_argument(
        "--parent-id",
        default="",
        help="Problem/parent id. Use with --rollout-id to form <parent_id>__rollout_<rollout_id>.",
    )
    parser.add_argument("--rollout-id", type=int, default=None, help="Rollout id used with --parent-id.")
    parser.add_argument(
        "--max-failed-nodes",
        type=int,
        default=0,
        help="Max failed nodes included per rollout. 0 means include all failed nodes. Default: 0.",
    )
    parser.add_argument(
        "--parent-context",
        choices=["direct", "ancestors", "none"],
        default="direct",
        help="Which parent facts to include for each failed node. Default: direct.",
    )
    parser.add_argument(
        "--include-raw-cot",
        action="store_true",
        help="Include input.raw_cot in the LLM prompt. Off by default because it can be large.",
    )
    parser.add_argument(
        "--include-full-graph",
        action="store_true",
        help="Include the full record.graph payload in the LLM prompt. Off by default because it can be large.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=8000,
        help="Max chars kept for long text/error/conversation fields. Default: 1200.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sampled prompt payloads without calling the API.",
    )
    parser.add_argument(
        "--export-sampled-rollout",
        action="store_true",
        help=(
            "Also export selected record_ids as a new rollout_flat.parquet subset "
            "for rerunning step-proof on the sampled rollouts."
        ),
    )
    parser.add_argument(
        "--source-rollout-name",
        default="",
        help=(
            "Source rollout name used with --export-sampled-rollout. "
            "Required when exporting, for example qwen3_8b_except_gsm8k."
        ),
    )
    parser.add_argument(
        "--sampled-rollout-name",
        default="",
        help=(
            "Output rollout name used with --export-sampled-rollout, without rollout_ prefix. "
            "Default: <source-rollout-name>_sampled_<exp_name>."
        ),
    )
    parser.add_argument(
        "--rollout-output-root",
        type=Path,
        default=None,
        help=(
            "Directory containing rollouts/ and step_proofs/. "
            "Default: parent of the inferred step_proofs root."
        ),
    )
    parser.add_argument(
        "--rollout-id-column",
        default="id",
        help="Record id column in rollout_flat.parquet. Default: id.",
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument(
        "--api-key",
        default="",
        help=(
            "API key passed directly on the command line. If omitted, the script reads "
            "--api-key-env instead."
        ),
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    return parser.parse_args()


def normalize_exp_name(name: str) -> str:
    name = str(name).strip()
    return name if name.startswith("step_proof_") else f"step_proof_{name}"


def infer_step_proof_root(project_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    project_root = project_root.expanduser().resolve()
    candidates = [
        project_root / "czx_work" / "step-proof" / "rollout_rerank_math_verify" / "outputs" / "step_proofs",
        project_root.parent / "czx_work" / "step-proof" / "rollout_rerank_math_verify" / "outputs" / "step_proofs",
    ]
    if step_proofs_root is not None:
        try:
            candidates.append(step_proofs_root())
        except Exception:
            pass

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def exp_dir_from_name(step_proof_root: Path, exp_name: str) -> Path:
    path = step_proof_root / normalize_exp_name(exp_name)
    if not path.is_dir():
        raise SystemExit(f"experiment not found: {path}")
    return path


def read_jsonl(path: Path) -> Iterable[JsonDict]:
    if not path.is_file():
        raise SystemExit(f"required file not found: {path}")
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def record_id(record: JsonDict) -> str:
    return str((record.get("meta") or {}).get("record_id") or "").strip()


def parent_id_from_record_id(rid: str) -> str:
    return rid.rsplit("__rollout_", 1)[0] if "__rollout_" in rid else rid


def rollout_id_from_record_id(rid: str) -> int | None:
    if "__rollout_" not in rid:
        return None
    try:
        return int(rid.rsplit("__rollout_", 1)[-1])
    except ValueError:
        return None


def target_record_id(args: argparse.Namespace) -> str:
    if args.record_id:
        return args.record_id.strip()
    if args.parent_id and args.rollout_id is not None:
        return f"{args.parent_id.strip()}__rollout_{args.rollout_id}"
    if args.parent_id or args.rollout_id is not None:
        raise SystemExit("--parent-id and --rollout-id must be provided together")
    return ""


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def truncate_text(value: Any, max_chars: int) -> str:
    text = text_value(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_skipped_derived(fact: JsonDict) -> bool:
    return (
        str(fact.get("origin") or "").strip().lower() == "derived"
        and int_value(fact.get("skip"), 0) == 1
    )


def is_prove_required_fact(fact: JsonDict, *, include_skipped_derived: bool) -> bool:
    origin = str(fact.get("origin") or "").strip().lower()
    if origin not in {"derived", "answer"}:
        return False
    return include_skipped_derived or not is_skipped_derived(fact)


def is_form_success_prove_failure(fact: JsonDict, *, include_skipped_derived: bool) -> bool:
    if not is_prove_required_fact(fact, include_skipped_derived=include_skipped_derived):
        return False
    if str(fact.get("form_status") or "").strip() != "success":
        return False
    formalization = fact.get("formalization") or {}
    if formalization.get("skipped") or not bool(formalization.get("lean_pass")):
        return False
    solved = fact.get("solved_lemma") or {}
    if solved.get("skipped"):
        return False
    prove_status = str(fact.get("prove_status") or "").strip()
    return prove_status != "success" or not bool(solved.get("lean_verify"))


def failed_facts(record: JsonDict, *, include_skipped_derived: bool) -> list[JsonDict]:
    facts = (record.get("results") or {}).get("facts") or []
    return [
        fact
        for fact in facts
        if isinstance(fact, dict)
        and is_form_success_prove_failure(
            fact,
            include_skipped_derived=include_skipped_derived,
        )
    ]


def prove_ratio_percent(record: JsonDict, *, include_skipped_derived: bool) -> float:
    facts = (record.get("results") or {}).get("facts") or []
    required = [
        fact
        for fact in facts
        if isinstance(fact, dict)
        and is_prove_required_fact(
            fact,
            include_skipped_derived=include_skipped_derived,
        )
    ]
    if not required:
        return 0.0
    verified = sum(1 for fact in required if bool((fact.get("solved_lemma") or {}).get("lean_verify")))
    return (verified / len(required)) * 100.0


def prove_ratio_bucket(percent: float, *, step: int) -> str:
    step = max(1, min(100, int(step)))
    clipped = max(0.0, min(100.0, percent))
    if clipped >= 100.0:
        return "100%"
    left = int(clipped // step) * step
    right = min(left + step, 100)
    return f"{left}-{right}%"


def prove_ratio_bucket_order(*, step: int) -> list[str]:
    step = max(1, min(100, int(step)))
    return [f"{left}-{min(left + step, 100)}%" for left in range(0, 100, step)] + ["100%"]


def compact_fact(
    fact: JsonDict,
    *,
    max_text_chars: int,
    include_formalization: bool,
    include_proof: bool,
) -> JsonDict:
    payload: JsonDict = {
        "fact_id": fact.get("fact_id", ""),
        "text": truncate_text(fact.get("text", ""), max_text_chars),
        "origin": fact.get("origin", ""),
        "parent_fact_ids": fact.get("parent_fact_ids") or [],
        "is_final_answer": bool(fact.get("is_final_answer")),
        "skip": fact.get("skip", 0),
        "proof_obligation": fact.get("proof_obligation") or {},
        "form_status": fact.get("form_status", ""),
        "prove_status": fact.get("prove_status", ""),
    }
    formalization = fact.get("formalization") or {}
    solved = fact.get("solved_lemma") or {}
    if include_formalization:
        payload["formalization"] = {
            "lean_code": truncate_text(formalization.get("lean_code", ""), max_text_chars * 2),
            "lean_pass": bool(formalization.get("lean_pass")),
            "error_msg": truncate_text(formalization.get("error_msg", ""), max_text_chars),
            "tries": formalization.get("tries", 0),
            "skipped": bool(formalization.get("skipped")),
        }
    else:
        payload["formalization_summary"] = {
            "lean_pass": bool(formalization.get("lean_pass")),
            "tries": formalization.get("tries", 0),
            "skipped": bool(formalization.get("skipped")),
        }
    if include_proof:
        payload["solved_lemma"] = {
            "lean_code": truncate_text(solved.get("lean_code", ""), max_text_chars * 2),
            "lean_pass": bool(solved.get("lean_pass")),
            "lean_verify": bool(solved.get("lean_verify")),
            "error_msg": truncate_text(solved.get("error_msg", ""), max_text_chars * 2),
            "tries": solved.get("tries", 0),
            "skipped": bool(solved.get("skipped")),
            "conversation_tail": compact_conversation_tail(
                solved.get("conversation_raw") or solved.get("conversation") or [],
                max_text_chars=max_text_chars,
            ),
        }
    else:
        payload["solved_lemma_summary"] = {
            "lean_verify": bool(solved.get("lean_verify")),
            "tries": solved.get("tries", 0),
            "skipped": bool(solved.get("skipped")),
            "error_msg": truncate_text(solved.get("error_msg", ""), max_text_chars),
        }
    return payload


def compact_conversation_tail(conversation: Any, *, max_text_chars: int) -> list[JsonDict]:
    if not isinstance(conversation, list):
        return []
    tail = conversation[-2:]
    rows: list[JsonDict] = []
    for msg in tail:
        if not isinstance(msg, dict):
            continue
        rows.append(
            {
                "role": msg.get("role", ""),
                "content": truncate_text(msg.get("content", ""), max_text_chars),
            }
        )
    return rows


def ancestor_ids(facts_by_id: dict[str, JsonDict], fact: JsonDict) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def visit(fid: str) -> None:
        if fid in seen:
            return
        seen.add(fid)
        parent = facts_by_id.get(fid)
        if not parent:
            return
        for grand in parent.get("parent_fact_ids") or []:
            visit(str(grand))
        ordered.append(fid)

    for parent_id in fact.get("parent_fact_ids") or []:
        visit(str(parent_id))
    return ordered


def build_rollout_payload(record: JsonDict, args: argparse.Namespace) -> JsonDict:
    rid = record_id(record)
    facts = (record.get("results") or {}).get("facts") or []
    facts_by_id = {str(fact.get("fact_id") or ""): fact for fact in facts if isinstance(fact, dict)}
    failures = failed_facts(record, include_skipped_derived=args.include_skipped_derived)
    if args.max_failed_nodes > 0:
        failures = failures[: args.max_failed_nodes]

    failed_nodes: list[JsonDict] = []
    for fact in failures:
        if args.parent_context == "none":
            parent_rows: list[JsonDict] = []
        elif args.parent_context == "ancestors":
            parent_rows = [
                compact_fact(
                    facts_by_id[parent_id],
                    max_text_chars=args.max_text_chars,
                    include_formalization=True,
                    include_proof=False,
                )
                for parent_id in ancestor_ids(facts_by_id, fact)
                if parent_id in facts_by_id
            ]
        else:
            parent_rows = [
                compact_fact(
                    facts_by_id[str(parent_id)],
                    max_text_chars=args.max_text_chars,
                    include_formalization=True,
                    include_proof=False,
                )
                for parent_id in fact.get("parent_fact_ids") or []
                if str(parent_id) in facts_by_id
            ]
        failed_nodes.append(
            {
                "node": compact_fact(
                    fact,
                    max_text_chars=args.max_text_chars,
                    include_formalization=True,
                    include_proof=True,
                ),
                "parents": parent_rows,
            }
        )

    input_payload = record.get("input") or {}
    graph = record.get("graph") or {}
    meta = record.get("meta") or {}
    payload: JsonDict = {
        "record_id": rid,
        "parent_id": parent_id_from_record_id(rid),
        "rollout_id": rollout_id_from_record_id(rid),
        "meta": {
            "graph_mode": meta.get("graph_mode"),
            "fdg_prompt": meta.get("fdg_prompt"),
            "formalizer_prompt": meta.get("formalizer_prompt"),
            "formalizer_context_mode": meta.get("formalizer_context_mode"),
            "prover_prompt": meta.get("prover_prompt"),
            "source_file": meta.get("source_file"),
            "source_row_pos": meta.get("source_row_pos"),
            "prove_verify_ratio_bucket": meta.get("prove_verify_ratio_bucket"),
            "prove_verify_ratio_bucket_step": meta.get("prove_verify_ratio_bucket_step"),
            "prove_verify_ratio_exclude_skipped_derived": meta.get(
                "prove_verify_ratio_exclude_skipped_derived"
            ),
        },
        "input": {
            "problem": truncate_text(input_payload.get("problem", ""), args.max_text_chars * 2),
        },
        "graph_context": {
            "validation_checks": meta.get("validation_checks") or {},
            "validation_warnings": graph.get("validation_warnings") or [],
            "final_fact_ids": graph.get("final_fact_ids") or [],
            "topo_order": graph.get("topo_order") or [],
        },
        "failed_nodes": failed_nodes,
    }
    if args.include_raw_cot:
        payload["input"]["raw_cot"] = truncate_text(input_payload.get("raw_cot", ""), args.max_text_chars * 3)
    if args.include_full_graph:
        payload["full_graph"] = graph
    return payload


def rollout_has_eligible_failures(record: JsonDict, args: argparse.Namespace) -> bool:
    return bool(failed_facts(record, include_skipped_derived=args.include_skipped_derived))


def select_stratified_records(stage3_path: Path, args: argparse.Namespace) -> list[JsonDict]:
    buckets: dict[str, list[JsonDict]] = {
        key: [] for key in prove_ratio_bucket_order(step=args.prove_ratio_bucket_step)
    }
    bucket_for_record: dict[str, str] = {}

    for rec in read_jsonl(stage3_path):
        if not rollout_has_eligible_failures(rec, args):
            continue
        ratio = prove_ratio_percent(
            rec,
            include_skipped_derived=args.include_skipped_derived,
        )
        bucket = prove_ratio_bucket(ratio, step=args.prove_ratio_bucket_step)
        buckets.setdefault(bucket, []).append(rec)
        rid = record_id(rec)
        if rid:
            bucket_for_record[rid] = bucket

    rng = random.Random(args.seed)
    for rows in buckets.values():
        rng.shuffle(rows)

    selected: list[JsonDict] = []
    per_bucket = max(0, int(args.samples_per_bucket))
    if per_bucket > 0:
        for bucket in prove_ratio_bucket_order(step=args.prove_ratio_bucket_step):
            selected.extend(buckets.get(bucket, [])[:per_bucket])
    else:
        limit = max(0, int(args.limit))
        if limit == 0:
            return []
        ordered_buckets = [
            bucket
            for bucket in prove_ratio_bucket_order(step=args.prove_ratio_bucket_step)
            if buckets.get(bucket)
        ]
        offsets = {bucket: 0 for bucket in ordered_buckets}
        while len(selected) < limit and ordered_buckets:
            progressed = False
            for bucket in ordered_buckets:
                offset = offsets[bucket]
                rows = buckets[bucket]
                if offset >= len(rows):
                    continue
                selected.append(rows[offset])
                offsets[bucket] = offset + 1
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break

    for rec in selected:
        meta = rec.setdefault("meta", {})
        if isinstance(meta, dict):
            rid = record_id(rec)
            meta["prove_verify_ratio_bucket"] = bucket_for_record.get(rid, "")
            meta["prove_verify_ratio_bucket_step"] = args.prove_ratio_bucket_step
            meta["prove_verify_ratio_exclude_skipped_derived"] = not args.include_skipped_derived
    return selected


def select_records(stage3_path: Path, args: argparse.Namespace) -> list[JsonDict]:
    exact_id = target_record_id(args)
    if exact_id:
        for rec in read_jsonl(stage3_path):
            if record_id(rec) == exact_id:
                return [rec] if rollout_has_eligible_failures(rec, args) else []
        raise SystemExit(f"record_id not found: {exact_id}")

    limit = max(0, int(args.limit))
    if limit == 0:
        return []

    if args.sample_mode == "stratified-random":
        return select_stratified_records(stage3_path, args)

    if args.sample_mode == "first":
        selected: list[JsonDict] = []
        for rec in read_jsonl(stage3_path):
            if rollout_has_eligible_failures(rec, args):
                selected.append(rec)
                if len(selected) >= limit:
                    break
        return selected

    rng = random.Random(args.seed)
    selected = []
    seen = 0
    for rec in read_jsonl(stage3_path):
        if not rollout_has_eligible_failures(rec, args):
            continue
        seen += 1
        if len(selected) < limit:
            selected.append(rec)
            continue
        idx = rng.randrange(seen)
        if idx < limit:
            selected[idx] = rec
    return selected


def build_messages(payload: JsonDict) -> list[JsonDict]:
    system = (
        "你是 Lean 4 / Mathlib / 数学证明调试助手。"
        "你的任务是分析已经 formalization 成功、但 prover 失败的 DAG 节点。"
        "请用中文自然语言回答，重点解释失败原因，不要编造未给出的运行结果。"
    )
    user = (
        "下面是一个 rollout 的 DAG 证明结果。failed_nodes 中的每个节点都满足："
        "form_status=success 且 formalization.lean_pass=true，但 prove 阶段失败或 lean_verify=false。\n\n"
        "请对 failed_nodes 里的每个节点逐一分析为什么出错。"
        "回答请使用 JSON，顶层字段为 record_id 和 node_reasons。"
        "node_reasons 是数组，每项包含 fact_id、reason_zh、evidence、likely_category。"
        "reason_zh 必须是中文自然语言；evidence 引用 Lean 错误、节点文本、父节点或 formalization 的具体证据。"
        "likely_category 可从以下类别中选：formal_statement_too_strong, missing_or_wrong_hypothesis, "
        "bad_parent_context, theorem_false_or_underspecified, prover_generated_invalid_code, "
        "prover_empty_or_truncated_output, lean_tactic_or_type_error, upstream_graph_issue, other。\n\n"
        f"输入 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_chat_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[JsonDict],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> JsonDict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(max(0, retries) + 1):
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            response_message = ((data.get("choices") or [{}])[0].get("message") or {})
            return {
                "ok": True,
                "content": content,
                "message": response_message,
                "raw_response": data,
                "duration_seconds": time.time() - started,
                "attempt": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(max(0.0, retry_sleep))
    return {
        "ok": False,
        "content": "",
        "error": last_error,
        "attempt": max(0, retries) + 1,
    }


def append_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_sampled_rollout(records: list[JsonDict], args: argparse.Namespace, step_proof_root: Path) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "--export-sampled-rollout requires pandas and a parquet engine such as pyarrow."
        ) from exc

    record_ids = [record_id(rec) for rec in records if record_id(rec)]
    if not record_ids:
        raise SystemExit("No selected record_id values to export as sampled rollout.")

    source_rollout_name = str(args.source_rollout_name or "").strip()
    if not source_rollout_name:
        raise SystemExit("--source-rollout-name is required with --export-sampled-rollout")

    sampled_rollout_name = str(args.sampled_rollout_name or "").strip()
    if not sampled_rollout_name:
        sampled_rollout_name = f"{source_rollout_name}_sampled_{normalize_exp_name(args.exp_name)}"

    output_root = (
        args.rollout_output_root.expanduser().resolve()
        if args.rollout_output_root is not None
        else step_proof_root.parent
    )
    source_path = output_root / "rollouts" / f"rollout_{source_rollout_name}" / "rollout_flat.parquet"
    if not source_path.is_file():
        raise SystemExit(f"source rollout parquet not found: {source_path}")

    id_column = str(args.rollout_id_column or "id")
    df = pd.read_parquet(source_path)
    if id_column not in df.columns:
        raise SystemExit(
            f"source rollout parquet missing id column {id_column!r}; columns={list(df.columns)}"
        )

    requested = set(record_ids)
    sampled = df[df[id_column].astype(str).isin(requested)].copy()
    found = set(sampled[id_column].astype(str).tolist())
    missing = sorted(requested - found)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... (+{len(missing) - 10})"
        raise SystemExit(f"{len(missing)} sampled record_id(s) missing from source rollout: {preview}{suffix}")

    order = {rid: idx for idx, rid in enumerate(record_ids)}
    sampled["_sample_order"] = sampled[id_column].astype(str).map(order)
    sampled = sampled.sort_values("_sample_order").drop(columns=["_sample_order"])

    out_dir = output_root / "rollouts" / f"rollout_{sampled_rollout_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollout_flat.parquet"
    sampled.to_parquet(out_path, index=False)

    manifest = {
        "source_rollout_name": source_rollout_name,
        "sampled_rollout_name": sampled_rollout_name,
        "source_path": str(source_path),
        "output_path": str(out_path),
        "record_count": int(len(sampled)),
        "record_ids": record_ids,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "limit": args.limit,
        "samples_per_bucket": args.samples_per_bucket,
        "prove_ratio_bucket_step": args.prove_ratio_bucket_step,
        "exclude_skipped_derived": not args.include_skipped_derived,
    }
    with (out_dir / "sample_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Exported sampled rollout: {out_path} ({len(sampled)} row(s))")
    print(f"Sampled rollout name: {sampled_rollout_name}")
    return out_path


def main() -> None:
    args = parse_args()
    step_proof_root = infer_step_proof_root(args.project_root, args.step_proof_root)
    exp_dir = exp_dir_from_name(step_proof_root, args.exp_name)
    stage3_path = exp_dir / "step_proof_results" / "result_stage3" / "stage3_results.jsonl"
    output = args.output or (exp_dir / "analysis" / "prove_failure_llm_reasons.jsonl")

    records = select_records(stage3_path, args)
    if not records:
        print("No eligible rollout records found.")
        return

    if args.export_sampled_rollout:
        export_sampled_rollout(records, args, step_proof_root)

    api_key = ""
    if not args.dry_run:
        api_key = str(args.api_key or "").strip() or os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                "API key is missing. Pass --api-key directly, or set the environment "
                f"variable named by --api-key-env ({args.api_key_env})."
            )

    rows: list[JsonDict] = []
    for idx, rec in enumerate(records, 1):
        payload = build_rollout_payload(rec, args)
        messages = build_messages(payload)
        print(
            f"[{idx}/{len(records)}] analyze {payload['record_id']} "
            f"failed_nodes={len(payload['failed_nodes'])}"
        )
        if args.dry_run:
            result = {
                "ok": True,
                "content": "",
                "message": {"role": "assistant", "content": ""},
                "dry_run": True,
            }
        else:
            result = call_chat_api(
                base_url=args.api_base_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
        rows.append(
            {
                "record_id": payload["record_id"],
                "parent_id": payload["parent_id"],
                "rollout_id": payload["rollout_id"],
                "failed_node_count": len(payload["failed_nodes"]),
                "request_payload": payload,
                "request_messages": messages,
                "response_message": result.get("message") or {
                    "role": "assistant",
                    "content": result.get("content", ""),
                },
                "conversation_messages": messages
                + [
                    result.get("message")
                    or {
                        "role": "assistant",
                        "content": result.get("content", ""),
                    }
                ],
                "llm_result": result,
            }
        )
        append_jsonl(output, [rows[-1]])

    print(f"Wrote {len(rows)} row(s) to {output}")


if __name__ == "__main__":
    main()


# python step-proof/experiments/rollout_rerank_math_verify/scripts/analyze_prove_failures_with_llm.py  \
# D:/program/research EXP_NAME  \
# --include-raw-cot --include-full-graph



# python step-proof/experiments/rollout_rerank_math_verify/scripts/analyze_prove_failures_with_llm.py \
#     D:/program/research ctx_c0_form_api \
#     --sample-mode stratified-random \
#     --samples-per-bucket 4 --seed 0 \
#     --export-sampled-rollout \
#     --source-rollout-name qwen3_8b_except_gsm8k \
#     --sampled-rollout-name qwen3_8b_except_gsm8k_sampled_ctx_c0_form_api_seed0 \
#     --rollout-id-column id

# python step-proof/experiments/rollout_rerank_math_verify/scripts/analyze_prove_failures_with_llm.py D:/program/research ctx_c0_form_api --sample-mode stratified-random --samples-per-bucket 4 --seed 0 --export-sampled-rollout --source-rollout-name qwen3_8b_except_gsm8k --sampled-rollout-name qwen3_8b_except_gsm8k_sampled_ctx_c0_form_api_seed0 --rollout-id-column id
