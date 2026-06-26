#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from path_defaults import step_proofs_root
except Exception:  # pragma: no cover
    step_proofs_root = None  # type: ignore[assignment]


JsonDict = dict[str, Any]
CATEGORIES = ["Number Theory", "Discrete & Prob", "Calculus", "Geometry", "Algebra", "UNKNOWN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize prove success ratios by LLM-classified problem category for one step-proof experiment."
        )
    )
    parser.add_argument("exp_name", help="Experiment name, with or without step_proof_ prefix.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\program\research"),
        help="Project root used to infer the step-proof output root. Default: D:/program/research",
    )
    parser.add_argument(
        "--step-proof-root",
        type=Path,
        default=None,
        help="Directory containing step_proof_* experiments. If omitted, inferred from --project-root.",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=None,
        help="Problem category JSONL. Default: <exp_dir>/analysis/problem_categories.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Default: <exp_dir>/analysis/prove_success_by_category.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--include-skipped-derived",
        action="store_true",
        help=(
            "Include derived nodes with skip=1 in the prove-node denominator. "
            "Default excludes skipped derived nodes, matching exclude_skipped_derived stats."
        ),
    )
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


def load_categories(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_jsonl(path):
        parent_id = str(row.get("parent_id") or "").strip()
        category = str(row.get("category") or "UNKNOWN").strip() or "UNKNOWN"
        if category not in CATEGORIES:
            category = "UNKNOWN"
        if parent_id:
            mapping[parent_id] = category
    return mapping


def required_prove_facts(record: JsonDict, *, include_skipped_derived: bool) -> list[JsonDict]:
    facts = (record.get("results") or {}).get("facts") or []
    required: list[JsonDict] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        origin = str(fact.get("origin") or "").strip().lower()
        if origin not in {"derived", "answer"}:
            continue
        if (
            not include_skipped_derived
            and origin == "derived"
            and int(fact.get("skip", 0)) == 1
        ):
            continue
        required.append(fact)
    return required


def fact_prove_success(fact: JsonDict) -> bool:
    solved = fact.get("solved_lemma") or {}
    return bool(solved.get("lean_verify"))


def empty_bucket(category: str) -> JsonDict:
    return {
        "category": category,
        "problems": set(),
        "rollouts": 0,
        "rollouts_with_required_nodes": 0,
        "rollouts_all_required_prove_success": 0,
        "prove_required_nodes": 0,
        "prove_success_nodes": 0,
    }


def pct(numer: int, denom: int) -> float:
    return round((numer / denom) * 100.0, 6) if denom else 0.0


def summarize(stage3_path: Path, category_by_parent: dict[str, str], *, include_skipped_derived: bool) -> list[JsonDict]:
    buckets: dict[str, JsonDict] = {category: empty_bucket(category) for category in CATEGORIES}

    for rec in read_jsonl(stage3_path):
        rid = record_id(rec)
        parent_id = parent_id_from_record_id(rid)
        category = category_by_parent.get(parent_id, "UNKNOWN")
        if category not in buckets:
            category = "UNKNOWN"
        bucket = buckets[category]
        bucket["problems"].add(parent_id)
        bucket["rollouts"] += 1

        required = required_prove_facts(rec, include_skipped_derived=include_skipped_derived)
        if required:
            bucket["rollouts_with_required_nodes"] += 1
        success = sum(1 for fact in required if fact_prove_success(fact))
        bucket["prove_required_nodes"] += len(required)
        bucket["prove_success_nodes"] += success
        if required and success == len(required):
            bucket["rollouts_all_required_prove_success"] += 1

    rows: list[JsonDict] = []
    for category in CATEGORIES:
        bucket = buckets[category]
        required_nodes = int(bucket["prove_required_nodes"])
        success_nodes = int(bucket["prove_success_nodes"])
        rollouts_with_required = int(bucket["rollouts_with_required_nodes"])
        rollout_success = int(bucket["rollouts_all_required_prove_success"])
        rows.append(
            {
                "category": category,
                "problem_count": len(bucket["problems"]),
                "rollout_count": bucket["rollouts"],
                "rollouts_with_required_nodes": rollouts_with_required,
                "prove_required_nodes": required_nodes,
                "prove_success_nodes": success_nodes,
                "node_prove_success_ratio": pct(success_nodes, required_nodes),
                "rollouts_all_required_prove_success": rollout_success,
                "rollout_all_required_prove_success_ratio": pct(rollout_success, rollouts_with_required),
            }
        )
    return rows


def write_csv(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "problem_count",
        "rollout_count",
        "rollouts_with_required_nodes",
        "prove_required_nodes",
        "prove_success_nodes",
        "node_prove_success_ratio",
        "rollouts_all_required_prove_success",
        "rollout_all_required_prove_success_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    step_proof_root = infer_step_proof_root(args.project_root, args.step_proof_root)
    exp_dir = exp_dir_from_name(step_proof_root, args.exp_name)
    stage3_path = exp_dir / "step_proof_results" / "result_stage3" / "stage3_results.jsonl"
    category_path = args.categories or (exp_dir / "analysis" / "problem_categories.jsonl")
    output_csv = args.output_csv or (exp_dir / "analysis" / "prove_success_by_category.csv")

    category_by_parent = load_categories(category_path)
    rows = summarize(
        stage3_path,
        category_by_parent,
        include_skipped_derived=args.include_skipped_derived,
    )
    write_csv(output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['category']}: nodes {row['prove_success_nodes']}/{row['prove_required_nodes']} "
            f"= {row['node_prove_success_ratio']:.2f}%; "
            f"rollouts {row['rollouts_all_required_prove_success']}/{row['rollouts_with_required_nodes']} "
            f"= {row['rollout_all_required_prove_success_ratio']:.2f}%"
        )
    print(f"Wrote CSV to {output_csv}")


if __name__ == "__main__":
    main()
