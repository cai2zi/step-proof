#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FLAT = EXP_DIR / "outputs" / "rollouts" / "rollout_qwen3_8b" / "rollout_flat.parquet"
DEFAULT_OUT_ROOT = EXP_DIR / "outputs" / "rollouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one rollout record, or one parent sample's rollouts, into a "
            "debug rollout_flat parquet/jsonl directory for the step-proof pipeline."
        )
    )
    parser.add_argument(
        "sample",
        help=(
            "Full rollout id (for example aime_2024__70__rollout_1) or parent sample id "
            "(for example aime_2024__70)."
        ),
    )
    parser.add_argument(
        "--source-flat",
        type=Path,
        default=DEFAULT_SOURCE_FLAT,
        help=f"Source rollout_flat parquet/jsonl. Default: {DEFAULT_SOURCE_FLAT}",
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        default=1,
        help="Rollout id to select when SAMPLE is a parent_id. Ignored with --all-rollouts.",
    )
    parser.add_argument(
        "--all-rollouts",
        action="store_true",
        help="When SAMPLE is a parent_id, keep every rollout for that parent sample.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Directory that contains rollout_* output dirs. Default: {DEFAULT_OUT_ROOT}",
    )
    parser.add_argument(
        "--out-name",
        default="",
        help=(
            "Output rollout name. If it does not start with rollout_, the prefix is added. "
            "Default: debug_<sample>[_rN|_all]."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing debug rollout_flat parquet/jsonl.",
    )
    parser.add_argument(
        "--contains",
        action="store_true",
        help="Allow substring matching if exact id/parent_id lookup finds nothing.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value[:120] or "sample"


def rollout_dir_name(name: str) -> str:
    return name if name.startswith("rollout_") else f"rollout_{name}"


def load_flat(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"source flat file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise SystemExit(f"unsupported source flat suffix: {path.suffix}")

    required = ["id", "parent_id", "rollout_id", "source", "question", "response", "gold"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing required column(s): {missing}; have {list(df.columns)}")
    return df


def _sort_rollouts(df: pd.DataFrame) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_rollout_id_sort"] = pd.to_numeric(sortable["rollout_id"], errors="coerce")
    sortable = sortable.sort_values(["parent_id", "_rollout_id_sort", "id"], kind="stable")
    return sortable.drop(columns=["_rollout_id_sort"])


def _suggest_matches(df: pd.DataFrame, sample: str, limit: int = 12) -> list[str]:
    ids = df["id"].astype(str)
    parents = df["parent_id"].astype(str)
    mask = ids.str.contains(sample, regex=False, na=False) | parents.str.contains(sample, regex=False, na=False)
    suggestions = _sort_rollouts(df.loc[mask, ["id", "parent_id", "rollout_id"]].drop_duplicates()).head(limit)
    return [
        f"{row.id}  (parent_id={row.parent_id}, rollout_id={row.rollout_id})"
        for row in suggestions.itertuples(index=False)
    ]


def select_rows(df: pd.DataFrame, sample: str, rollout_id: int, all_rollouts: bool, contains: bool) -> pd.DataFrame:
    work = df.copy()
    work["id"] = work["id"].astype(str)
    work["parent_id"] = work["parent_id"].astype(str)

    by_id = work[work["id"] == sample]
    if not by_id.empty:
        return by_id.reset_index(drop=True)

    by_parent = work[work["parent_id"] == sample]
    if not by_parent.empty:
        if all_rollouts:
            return _sort_rollouts(by_parent).reset_index(drop=True)
        picked = by_parent[pd.to_numeric(by_parent["rollout_id"], errors="coerce") == rollout_id]
        if picked.empty:
            available = sorted(
                {
                    int(value)
                    for value in pd.to_numeric(by_parent["rollout_id"], errors="coerce").dropna().tolist()
                }
            )
            raise SystemExit(
                f"parent_id={sample!r} exists, but rollout_id={rollout_id} was not found. "
                f"Available rollout_id values: {available}"
            )
        return _sort_rollouts(picked).reset_index(drop=True)

    if contains:
        ids = work["id"].str.contains(sample, regex=False, na=False)
        parents = work["parent_id"].str.contains(sample, regex=False, na=False)
        matched = work[ids | parents]
        if not matched.empty:
            if all_rollouts:
                return _sort_rollouts(matched).reset_index(drop=True)
            picked = matched[pd.to_numeric(matched["rollout_id"], errors="coerce") == rollout_id]
            if not picked.empty:
                return _sort_rollouts(picked).reset_index(drop=True)
            return _sort_rollouts(matched.head(1)).reset_index(drop=True)

    suggestions = _suggest_matches(work, sample)
    msg = f"no rollout row found for id/parent_id: {sample!r}"
    if suggestions:
        msg += "\nSimilar rows:\n  " + "\n  ".join(suggestions)
    raise SystemExit(msg)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_out_name(sample: str, selected: pd.DataFrame, all_rollouts: bool, rollout_id: int) -> str:
    if len(selected) == 1:
        stem = sanitize_name(str(selected.iloc[0]["id"]))
        return f"debug_{stem}"
    else:
        parent_ids = sorted({str(value) for value in selected["parent_id"].tolist()})
        stem = sanitize_name(parent_ids[0] if len(parent_ids) == 1 else sample)
    suffix = "all" if all_rollouts and len(selected) > 1 else f"r{rollout_id}"
    return f"debug_{stem}_{suffix}"


def main() -> None:
    args = parse_args()
    df = load_flat(args.source_flat)
    selected = select_rows(
        df,
        sample=str(args.sample),
        rollout_id=int(args.rollout_id),
        all_rollouts=bool(args.all_rollouts),
        contains=bool(args.contains),
    )

    out_name = args.out_name.strip() or default_out_name(args.sample, selected, args.all_rollouts, args.rollout_id)
    out_dir = args.out_root / rollout_dir_name(out_name)
    parquet_path = out_dir / "rollout_flat.parquet"
    jsonl_path = out_dir / "rollout_flat.jsonl"
    manifest_path = out_dir / "debug_manifest.json"

    if out_dir.exists() and not args.force and (parquet_path.exists() or jsonl_path.exists()):
        raise SystemExit(f"output already exists: {out_dir} (pass --force to overwrite)")

    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(parquet_path, index=False)
    rows = selected.to_dict("records")
    write_jsonl(jsonl_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "source_flat": str(args.source_flat),
                "sample": args.sample,
                "rollout_id": args.rollout_id,
                "all_rollouts": bool(args.all_rollouts),
                "rows": len(rows),
                "ids": [str(row["id"]) for row in rows],
                "parent_ids": sorted({str(row["parent_id"]) for row in rows}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] selected {len(rows)} row(s)")
    print(f"[done] parquet -> {parquet_path}")
    print(f"[done] jsonl    -> {jsonl_path}")
    print(f"[done] manifest -> {manifest_path}")
    print()
    print("Use these overrides for step-proof debug:")
    print(f"  stage1.parquet_dir={out_dir}")
    print("  stage1.parquet_glob=rollout_flat.parquet")
    print("  stage1.limit=-1")
    print("  stage2.limit=-1")
    print("  stage3.limit=-1")

    rollout_name = out_dir.name.removeprefix("rollout_")
    if args.out_root.resolve() == DEFAULT_OUT_ROOT.resolve():
        print()
        print("Or, with the existing experiment output_root:")
        print(f"  rollout_name={rollout_name}")


if __name__ == "__main__":
    main()
