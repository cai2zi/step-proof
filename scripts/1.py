#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP_PROOFS_ROOT = (
    PROJECT_ROOT
    / "czx_work"
    / "step-proof"
    / "rollout_rerank_math_verify"
    / "outputs"
    / "step_proofs"
)
STATS_REL_PATH = Path("step_proof_results") / "stats" / "stage3_verify_stats.json"
TARGET_SECTION = "exclude_skipped_derived"
TARGET_FIELDS = (
    "global_prove_verified_nodes_ratio",
    "global_form_verified_nodes_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate stage3 exclude_skipped_derived global verify ratios "
            "from step_proof experiment outputs into a CSV."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_STEP_PROOFS_ROOT,
        help=f"Directory containing step_proof_* experiment folders. Default: {DEFAULT_STEP_PROOFS_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/stage3_verify_ratios.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any experiment is missing the stats file or target fields.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def iter_experiment_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            yield child


def read_ratio_row(exp_dir: Path, strict: bool) -> Dict[str, Any] | None:
    stats_path = exp_dir / STATS_REL_PATH
    if not stats_path.exists():
        message = f"[skip] missing stats file: {stats_path}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    stats = load_json(stats_path)
    section = stats.get(TARGET_SECTION)
    if not isinstance(section, dict):
        message = f"[skip] missing object '{TARGET_SECTION}' in {stats_path}"
        if strict:
            raise KeyError(message)
        print(message)
        return None

    row: Dict[str, Any] = {"experiment": exp_dir.name}
    missing_fields: List[str] = []
    for field in TARGET_FIELDS:
        if field not in section:
            missing_fields.append(field)
            row[field] = ""
        else:
            row[field] = section[field]

    if missing_fields:
        message = f"[skip] missing fields {missing_fields} in {stats_path}"
        if strict:
            raise KeyError(message)
        print(message)
        return None

    return row


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "stage3_verify_ratios.csv")).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    rows = [
        row
        for row in (read_ratio_row(exp_dir, strict=args.strict) for exp_dir in iter_experiment_dirs(root))
        if row is not None
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", *TARGET_FIELDS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
