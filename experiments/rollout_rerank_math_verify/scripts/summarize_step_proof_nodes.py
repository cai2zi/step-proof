#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STEP_PROOF_ROOT = EXP_DIR / "outputs" / "step_proofs"


JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize FDG step-proof stage2/stage3 node stats for one experiment name. "
            "Reads final JSONL plus checkpoints by default, so unfinished runs are included."
        )
    )
    parser.add_argument(
        "name",
        help="Experiment name, with or without the step_proof_ prefix.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_STEP_PROOF_ROOT,
        help=f"Directory containing step_proof_* experiments. Default: {DEFAULT_STEP_PROOF_ROOT}",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Read only stage*_results.jsonl and ignore checkpoint files.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full summary JSON.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=12,
        help="Number of raw failed error examples to print per stage. Default: 12.",
    )
    return parser.parse_args()


def exp_dir_from_name(root: Path, name: str) -> Path:
    candidates = [name]
    if not name.startswith("step_proof_"):
        candidates.insert(0, f"step_proof_{name}")
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return root / candidates[0]


def read_jsonl(path: Path) -> Iterable[JsonDict]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def stage_paths(exp_dir: Path, stage: str) -> tuple[Path, Path]:
    stage_num = stage.removeprefix("stage")
    stage_dir = exp_dir / "step_proof_results" / f"result_stage{stage_num}"
    return stage_dir / f"stage{stage_num}_results.jsonl", stage_dir / f"stage{stage_num}_ckpt"


def record_id(record: JsonDict) -> str:
    return str((record.get("meta") or {}).get("record_id") or "").strip()


def facts_from_final(record: JsonDict) -> list[JsonDict]:
    return list((record.get("results") or {}).get("facts") or [])


def facts_from_checkpoint(record: JsonDict) -> list[JsonDict]:
    runtime_facts = record.get("runtime_facts") or {}
    if isinstance(runtime_facts, dict):
        return [dict(fact) for fact in runtime_facts.values()]
    return []


def load_stage_records(exp_dir: Path, stage: str, *, include_checkpoints: bool) -> list[tuple[str, JsonDict, bool]]:
    results_path, ckpt_dir = stage_paths(exp_dir, stage)
    records: dict[str, tuple[JsonDict, bool]] = {}

    for rec in read_jsonl(results_path) or []:
        rid = record_id(rec)
        if rid:
            records[rid] = (rec, False)

    if include_checkpoints and ckpt_dir.is_dir():
        for path in sorted(ckpt_dir.glob("*.json")):
            rec = read_json(path)
            rid = record_id(rec) or path.stem
            if rid not in records:
                records[rid] = (rec, True)

    return [(rid, rec, is_checkpoint) for rid, (rec, is_checkpoint) in sorted(records.items())]


def normalize_error_msg(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def classify_error(error_msg: str, payload: JsonDict, status: str) -> str:
    text = error_msg.lower()
    lean_code = str(payload.get("lean_code") or "")

    if status in {"pending", "running"}:
        return status
    if payload.get("skipped"):
        return "skipped"
    if "token_overflow" in text or "prompt token" in text or "token limit" in text:
        return "token_overflow"
    if "output_truncated" in text or "truncated" in text:
        return "output_truncated"
    if "no lean 4 code block" in text or "no lean declaration" in text or "extract" in text:
        return "extract_error"
    if "sorry" in text or "hassorry" in text or "declaration uses" in text:
        return "sorry_or_incomplete_proof"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "out of memory" in text or "cuda" in text or "memory" in text:
        return "runtime_memory"
    if "maximum recursion depth" in text or "recursion" in text:
        return "runtime_recursion"
    if '"severity": "error"' in text or '"severity":"error"' in text or '"severity": 1' in text or '"severity":1' in text:
        return lean_diagnostic_category(text)
    if "'severity': 'error'" in text:
        return lean_diagnostic_category(text)
    if "unknown identifier" in text:
        return "lean_unknown_identifier"
    if "unknown constant" in text:
        return "lean_unknown_constant"
    if "type mismatch" in text:
        return "lean_type_mismatch"
    if "application type mismatch" in text:
        return "lean_application_type_mismatch"
    if "failed to synthesize" in text:
        return "lean_synthesis_failed"
    if "tactic" in text and "failed" in text:
        return "lean_tactic_failed"
    if "unsolved goals" in text or "unsolved goal" in text:
        return "lean_unsolved_goals"
    if "invalid field" in text:
        return "lean_invalid_field"
    if "unexpected token" in text or "expected" in text:
        return "lean_syntax_or_parse"
    if lean_code:
        return "lean_error_other"
    if not error_msg:
        return "failed_no_error_msg"
    return "other_error"


def lean_diagnostic_category(text: str) -> str:
    if "unknown identifier" in text:
        return "lean_unknown_identifier"
    if "unknown constant" in text:
        return "lean_unknown_constant"
    if "type mismatch" in text:
        return "lean_type_mismatch"
    if "unsolved goals" in text or "unsolved goal" in text:
        return "lean_unsolved_goals"
    if "invalid field" in text:
        return "lean_invalid_field"
    if "application type mismatch" in text:
        return "lean_application_type_mismatch"
    if "unexpected token" in text or "expected" in text:
        return "lean_syntax_or_parse"
    if "failed to synthesize" in text:
        return "lean_synthesis_failed"
    return "lean_error_other"


def node_status(fact: JsonDict, stage: str) -> str:
    if stage == "stage2":
        return str(fact.get("form_status") or "").strip() or "unknown"
    return str(fact.get("prove_status") or "").strip() or "unknown"


def node_payload(fact: JsonDict, stage: str) -> JsonDict:
    if stage == "stage2":
        return dict(fact.get("formalization") or {})
    return dict(fact.get("solved_lemma") or {})


def node_success(fact: JsonDict, stage: str) -> bool:
    payload = node_payload(fact, stage)
    if stage == "stage2":
        return bool(payload.get("lean_pass"))
    return bool(payload.get("lean_verify"))


def summarize_stage(records: list[tuple[str, JsonDict, bool]], stage: str) -> JsonDict:
    total_nodes = 0
    origin_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    need_count = 0
    success_count = 0
    failed_count = 0
    error_counts: Counter[str] = Counter()
    error_examples: list[JsonDict] = []
    checkpoint_records = 0
    final_records = 0

    for rid, rec, is_checkpoint in records:
        checkpoint_records += int(is_checkpoint)
        final_records += int(not is_checkpoint)
        facts = facts_from_checkpoint(rec) if is_checkpoint else facts_from_final(rec)
        for fact in facts:
            total_nodes += 1
            origin = str(fact.get("origin") or "<missing>").strip() or "<missing>"
            origin_counts[origin] += 1

            status = node_status(fact, stage)
            status_counts[status] += 1
            if status == "skipped":
                continue

            need_count += 1
            if node_success(fact, stage):
                success_count += 1
                continue

            failed_count += 1
            payload = node_payload(fact, stage)
            error_msg = normalize_error_msg(payload.get("error_msg"))
            category = classify_error(error_msg, payload, status)
            error_counts[category] += 1
            if len(error_examples) < 100:
                error_examples.append(
                    {
                        "record_id": rid,
                        "fact_id": fact.get("fact_id"),
                        "origin": origin,
                        "status": status,
                        "category": category,
                        "error_msg": compact(error_msg),
                    }
                )

    skipped_count = status_counts.get("skipped", 0)
    return {
        "records": {
            "total": len(records),
            "final": final_records,
            "checkpoint": checkpoint_records,
        },
        "total_nodes": total_nodes,
        "origin_counts": dict(sorted(origin_counts.items())),
        "skipped_nodes": skipped_count,
        "need_nodes": need_count,
        "success_nodes": success_count,
        "failed_nodes": failed_count,
        "status_counts": dict(sorted(status_counts.items())),
        "failed_error_counts": dict(error_counts.most_common()),
        "failed_error_examples": error_examples,
    }


def compact(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_counter(title: str, counts: dict[str, int]) -> None:
    print(title)
    if not counts:
        print("  <none>")
        return
    width = max(len(str(key)) for key in counts)
    for key, value in counts.items():
        print(f"  {key:<{width}}  {value}")


def print_stage(stage: str, summary: JsonDict, *, show_errors: int) -> None:
    title = "Stage 2 / formalization" if stage == "stage2" else "Stage 3 / proving"
    print(f"\n== {title} ==")
    records = summary["records"]
    print(
        f"records: total={records['total']} final={records['final']} checkpoint={records['checkpoint']}"
    )
    print(f"1. total_nodes: {summary['total_nodes']}")
    print_counter("2. origin_counts:", summary["origin_counts"])
    print(f"3. skipped_nodes: {summary['skipped_nodes']}")
    print(f"4. need_{'form' if stage == 'stage2' else 'prove'}_nodes: {summary['need_nodes']}")
    print(f"5. success_nodes_in_need: {summary['success_nodes']}")
    print(f"6. failed_nodes_in_need: {summary['failed_nodes']}")
    print_counter("7. failed_error_counts:", summary["failed_error_counts"])
    print_counter("status_counts:", summary["status_counts"])

    examples = summary["failed_error_examples"][: max(0, show_errors)]
    if examples:
        print(f"failed_error_examples(first {len(examples)}):")
        for item in examples:
            print(
                f"  - {item['category']} | record={item['record_id']} "
                f"fact={item['fact_id']} origin={item['origin']} status={item['status']}"
            )
            if item["error_msg"]:
                print(f"    {item['error_msg']}")


def main() -> None:
    args = parse_args()
    exp_dir = exp_dir_from_name(args.root, args.name)
    if not exp_dir.is_dir():
        raise SystemExit(f"step-proof experiment not found: {exp_dir}")

    summary = {
        "experiment": exp_dir.name,
        "experiment_dir": str(exp_dir),
        "include_checkpoints": not args.final_only,
        "stages": {},
    }
    for stage in ("stage2", "stage3"):
        records = load_stage_records(exp_dir, stage, include_checkpoints=not args.final_only)
        summary["stages"][stage] = summarize_stage(records, stage)

    print(f"experiment: {summary['experiment']}")
    print(f"dir: {summary['experiment_dir']}")
    print(f"include_checkpoints: {summary['include_checkpoints']}")
    for stage in ("stage2", "stage3"):
        print_stage(stage, summary["stages"][stage], show_errors=args.show_errors)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\njson -> {args.json_out}")


if __name__ == "__main__":
    main()
