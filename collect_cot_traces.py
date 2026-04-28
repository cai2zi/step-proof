from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from proofflow.graph_mode import FDG_GRAPH_MODE, ensure_single_graph_mode

JsonDict = Dict[str, Any]


def _load_jsonl(path: Path) -> Iterable[JsonDict]:
    if not path.is_file():
        return []
    rows: List[JsonDict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _meta(rec: JsonDict) -> JsonDict:
    meta = rec.get("meta") or {}
    return {
        "record_id": str(meta.get("record_id", "")).strip(),
        "source_file": meta.get("source_file", ""),
        "source_row_pos": meta.get("source_row_pos"),
    }


def _item_base(rec: JsonDict, node: JsonDict, stage: str, graph_mode: str) -> JsonDict:
    base = _meta(rec)
    base["stage"] = stage
    base["graph_mode"] = graph_mode
    if graph_mode == FDG_GRAPH_MODE:
        base.update(
            {
                "fact_id": node.get("fact_id", ""),
                "text": node.get("text", ""),
                "parent_fact_ids": node.get("parent_fact_ids", []),
                "origin": node.get("origin", ""),
                "is_final_answer": node.get("is_final_answer", False),
                "proof_obligation": node.get("proof_obligation", {}),
            }
        )
    else:
        base.update(
            {
                "node_id": node.get("id", ""),
                "role": node.get("role", ""),
                "node_type": node.get("node_type", ""),
                "needs_verification": node.get("needs_verification"),
                "statement": node.get("statement", ""),
                "natural_language": node.get("natural_language", ""),
            }
        )
    return base


def _formal_rows(rows: Iterable[JsonDict], include_history: bool, graph_mode: str) -> Iterable[JsonDict]:
    key = "facts" if graph_mode == FDG_GRAPH_MODE else "nodes"
    for rec in rows:
        for node in rec.get("results", {}).get(key, []) or []:
            formalization = node.get("formalization") or {}
            out = _item_base(rec, node, "formal", graph_mode)
            out.update(
                {
                    "status": node.get("form_status", ""),
                    "tries": formalization.get("tries", 0),
                    "lean_pass": formalization.get("lean_pass", False),
                    "lean_code": formalization.get("lean_code", ""),
                    "error_msg": formalization.get("error_msg"),
                    "dependency_context_block": formalization.get(
                        "dependency_context_block", ""
                    ),
                }
            )
            if include_history:
                out["attempt_history"] = formalization.get("attempt_history", [])
            yield out


def _prove_rows(rows: Iterable[JsonDict], include_history: bool, graph_mode: str) -> Iterable[JsonDict]:
    key = "facts" if graph_mode == FDG_GRAPH_MODE else "nodes"
    for rec in rows:
        for node in rec.get("results", {}).get(key, []) or []:
            solved = node.get("solved_lemma") or {}
            out = _item_base(rec, node, "prove", graph_mode)
            out.update(
                {
                    "status": node.get("prove_status", ""),
                    "tries": solved.get("tries", 0),
                    "lean_pass": solved.get("lean_pass", False),
                    "lean_verify": solved.get("lean_verify", False),
                    "lean_code": solved.get("lean_code", ""),
                    "error_msg": solved.get("error_msg"),
                }
            )
            if include_history:
                out["attempt_history"] = solved.get("attempt_history", [])
            yield out


def _record_summary_rows(stage3_rows: Iterable[JsonDict], graph_mode: str) -> Iterable[JsonDict]:
    key = "facts" if graph_mode == FDG_GRAPH_MODE else "nodes"
    for rec in stage3_rows:
        nodes = rec.get("results", {}).get(key, []) or []
        summary = _meta(rec)
        summary.update(
            {
                "graph_mode": graph_mode,
                "record_status": (rec.get("execution") or {}).get("record_status", ""),
                "node_count": len(nodes),
                "form_success": sum(
                    1 for node in nodes if node.get("form_status") == "success"
                ),
                "form_failed": sum(
                    1 for node in nodes if node.get("form_status") == "failed"
                ),
                "prove_success": sum(
                    1 for node in nodes if node.get("prove_status") == "success"
                ),
                "prove_failed": sum(
                    1 for node in nodes if node.get("prove_status") == "failed"
                ),
            }
        )
        yield summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect last formal/prove attempt traces from stage outputs.",
    )
    parser.add_argument("--stage2-jsonl", type=Path, required=True)
    parser.add_argument("--stage3-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-attempt-history", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    stage2_rows = list(_load_jsonl(args.stage2_jsonl))
    stage3_rows = list(_load_jsonl(args.stage3_jsonl))
    stage2_graph_mode = ensure_single_graph_mode(stage2_rows, source_name=str(args.stage2_jsonl))
    stage3_graph_mode = ensure_single_graph_mode(stage3_rows, source_name=str(args.stage3_jsonl))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "formal_last_attempts": _append_jsonl(
            args.out_dir / "formal_last_attempts.jsonl",
            _formal_rows(stage2_rows, args.include_attempt_history, stage2_graph_mode),
        ),
        "prove_last_attempts": _append_jsonl(
            args.out_dir / "prove_last_attempts.jsonl",
            _prove_rows(stage3_rows, args.include_attempt_history, stage3_graph_mode),
        ),
        "record_summary": _append_jsonl(
            args.out_dir / "record_summary.jsonl",
            _record_summary_rows(stage3_rows, stage3_graph_mode),
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
