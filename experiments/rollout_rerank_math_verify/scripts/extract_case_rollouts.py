#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from path_defaults import default_rollout_flat, rollouts_root, step_proofs_root

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STEP_PROOF_ROOT = step_proofs_root()
DEFAULT_SOURCE_FLAT = default_rollout_flat("qwen3_8b")
DEFAULT_OUT_ROOT = rollouts_root()


JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine step-proof comparison cases and extract the corresponding rollout_flat "
            "parquet/jsonl for the next experiment."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help=(
            "One or two step-proof experiment names, with or without the step_proof_ prefix. "
            "With one experiment, A-vs-B buckets are omitted."
        ),
    )
    parser.add_argument(
        "--step-proof-root",
        type=Path,
        default=DEFAULT_STEP_PROOF_ROOT,
        help=f"Directory containing step_proof_* experiments. Default: {DEFAULT_STEP_PROOF_ROOT}",
    )
    parser.add_argument(
        "--source-flat",
        type=Path,
        default=DEFAULT_SOURCE_FLAT,
        help=f"Source rollout_flat parquet/jsonl. Default: {DEFAULT_SOURCE_FLAT}",
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
        help="Output rollout name for the combined extraction. Default is derived from experiments/mode.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help=(
            "Bucket(s) to extract. Can be repeated. Default: all non-empty buckets. "
            "Known buckets: pass4_correct_selected_wrong, pass1_correct_selected_wrong, "
            "high_score_wrong, a_wrong_b_correct, a_correct_b_wrong."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["all-rollouts", "selected", "correct", "selected-and-correct"],
        default="all-rollouts",
        help=(
            "Which rollout rows to extract for each case parent. "
            "Default all-rollouts keeps every candidate rollout for each mined parent."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max cases kept per bucket after sorting, matching the viewer default. Use -1 for no limit.",
    )
    parser.add_argument(
        "--write-per-bucket",
        action="store_true",
        help="Also write one rollout_* directory per bucket.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing rollout_flat parquet/jsonl files.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the mined case manifest JSON.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value[:140] or "cases"


def rollout_dir_name(name: str) -> str:
    return name if name.startswith("rollout_") else f"rollout_{name}"


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        raise SystemExit(f"required file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["parent_id"] = df["parent_id"].astype(str)
    return df


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool_correct(value: Any) -> bool:
    return str(value).strip() == "1"


def _parent_id(row: JsonDict) -> str:
    parent = str(row.get("parent_id") or "").strip()
    if parent:
        return parent
    rid = str(row.get("id") or "").strip()
    if "__rollout_" in rid:
        return rid.rsplit("__rollout_", 1)[0]
    return rid


def _rollout_id(row: JsonDict) -> int | None:
    if str(row.get("rollout_id", "")).strip():
        try:
            return int(row["rollout_id"])
        except (TypeError, ValueError):
            pass
    rid = str(row.get("id") or "")
    if "__rollout_" in rid:
        try:
            return int(rid.rsplit("__rollout_", 1)[-1])
        except ValueError:
            return None
    return None


def _score_sort_key(row: JsonDict) -> tuple:
    return (
        -_safe_float(row.get("success_ratio")),
        -_safe_int(row.get("prove_success_nodes")),
        -_safe_int(row.get("lean_pass_nodes")),
        _safe_int(row.get("rollout_id"), 10**9),
    )


def _score_rank_key(row: JsonDict) -> tuple:
    return (
        _safe_float(row.get("success_ratio")),
        _safe_int(row.get("prove_success_nodes")),
        _safe_int(row.get("lean_pass_nodes")),
        -_safe_int(row.get("rollout_id"), 10**9),
    )


def normalize_exp_name(name: str) -> str:
    name = str(name).strip()
    return name if name.startswith("step_proof_") else f"step_proof_{name}"


def exp_dir(root: Path, name: str) -> Path:
    normalized = normalize_exp_name(name)
    path = root / normalized
    if not path.is_dir():
        raise SystemExit(f"step-proof experiment not found: {path}")
    return path


class ExperimentData:
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.scores_by_parent: dict[str, list[JsonDict]] = {}
        self.scores_by_id: dict[str, JsonDict] = {}
        self.eval_by_rollout: dict[str, JsonDict] = {}
        self.selected_eval_by_parent: dict[str, JsonDict] = {}
        self.pass_at_1_by_parent: dict[str, bool] = {}
        self.pass_at_k_by_parent: dict[str, bool] = {}
        self.parent_source: dict[str, str] = {}


def load_experiment(root: Path, raw_name: str) -> ExperimentData:
    name = normalize_exp_name(raw_name)
    path = exp_dir(root, raw_name)
    data = ExperimentData(name, path)

    for row in read_jsonl(path / "scores.jsonl"):
        rid = str(row.get("id") or "").strip()
        parent = _parent_id(row)
        if not rid or not parent:
            continue
        row = dict(row)
        row["parent_id"] = parent
        row["rollout_id"] = _rollout_id(row)
        data.scores_by_parent.setdefault(parent, []).append(row)
        data.scores_by_id[rid] = row
        if row.get("source"):
            data.parent_source.setdefault(parent, str(row.get("source") or ""))
    for rows in data.scores_by_parent.values():
        rows.sort(key=_score_sort_key)

    for row in read_jsonl(path / "math_verify" / "all_rollouts_eval.jsonl"):
        rid = str(row.get("id") or "").strip()
        parent = _parent_id(row)
        if not rid or not parent:
            continue
        data.eval_by_rollout[rid] = row
        ok = _to_bool_correct(row.get("is_correct"))
        data.pass_at_k_by_parent[parent] = data.pass_at_k_by_parent.get(parent, False) or ok
        if _rollout_id(row) == 1:
            data.pass_at_1_by_parent[parent] = ok
        if row.get("source"):
            data.parent_source.setdefault(parent, str(row.get("source") or ""))

    for row in read_jsonl(path / "math_verify" / "step_proof_best_eval.jsonl"):
        parent = _parent_id(row)
        if parent:
            data.selected_eval_by_parent[parent] = row
            if row.get("source"):
                data.parent_source.setdefault(parent, str(row.get("source") or ""))
    return data


def selected_info(data: ExperimentData, parent_id: str) -> JsonDict:
    selected = data.selected_eval_by_parent.get(parent_id) or {}
    rollout = _rollout_id(selected)
    rid = f"{parent_id}__rollout_{rollout}" if rollout is not None else ""
    score = data.scores_by_id.get(rid, {})
    return {
        "rollout_id": rollout if rollout is not None else "",
        "record_id": rid,
        "is_correct": _to_bool_correct(selected.get("is_correct")),
        "success_ratio": score.get("success_ratio", ""),
        "prove_success_nodes": score.get("prove_success_nodes", ""),
        "lean_pass_nodes": score.get("lean_pass_nodes", ""),
    }


def case_payload(parent_id: str, exp_names: list[str], loaded: dict[str, ExperimentData]) -> JsonDict:
    first = loaded[exp_names[0]]
    first_row = (first.scores_by_parent.get(parent_id) or [{}])[0]
    by_exp = {name: selected_info(data, parent_id) for name, data in loaded.items()}
    correct_rollouts: dict[str, list[int]] = {}
    for name, data in loaded.items():
        correct_rollouts[name] = sorted(
            rollout
            for rid, row in data.eval_by_rollout.items()
            if _parent_id(row) == parent_id and _to_bool_correct(row.get("is_correct"))
            for rollout in [_rollout_id(row)]
            if rollout is not None
        )
    return {
        "parent_id": parent_id,
        "source": first.parent_source.get(parent_id, ""),
        "question": str(first_row.get("question") or "")[:240],
        "gold": first_row.get("gold", ""),
        "by_exp": by_exp,
        "correct_rollouts": correct_rollouts,
    }


def mine_cases(loaded: dict[str, ExperimentData], limit: int) -> JsonDict:
    exp_names = list(loaded)
    parent_sets = [set(data.scores_by_parent.keys()) for data in loaded.values()]
    common_parents = sorted(set.intersection(*parent_sets)) if parent_sets else []
    buckets: dict[str, list[JsonDict]] = {
        "pass4_correct_selected_wrong": [],
        "pass1_correct_selected_wrong": [],
        "high_score_wrong": [],
    }
    if len(exp_names) == 2:
        buckets["a_wrong_b_correct"] = []
        buckets["a_correct_b_wrong"] = []

    for parent in common_parents:
        selected_ok = {name: selected_info(data, parent)["is_correct"] for name, data in loaded.items()}
        pass4_ok = {name: data.pass_at_k_by_parent.get(parent, False) for name, data in loaded.items()}
        pass1_ok = {name: data.pass_at_1_by_parent.get(parent, False) for name, data in loaded.items()}
        all_selected_wrong = all(not value for value in selected_ok.values())

        if all(pass4_ok.values()) and all_selected_wrong:
            buckets["pass4_correct_selected_wrong"].append(case_payload(parent, exp_names, loaded))
        if all(pass1_ok.values()) and all_selected_wrong:
            buckets["pass1_correct_selected_wrong"].append(case_payload(parent, exp_names, loaded))
        if len(exp_names) == 2:
            a, b = exp_names
            if not selected_ok[a] and selected_ok[b]:
                buckets["a_wrong_b_correct"].append(case_payload(parent, exp_names, loaded))
            if selected_ok[a] and not selected_ok[b]:
                buckets["a_correct_b_wrong"].append(case_payload(parent, exp_names, loaded))

        if any(not ok for ok in selected_ok.values()):
            payload = case_payload(parent, exp_names, loaded)
            best_wrong_score = (-math.inf, -math.inf, -math.inf, -math.inf)
            for name, data in loaded.items():
                info = selected_info(data, parent)
                if info["is_correct"]:
                    continue
                score_row = data.scores_by_id.get(str(info.get("record_id") or ""), {})
                best_wrong_score = max(best_wrong_score, _score_rank_key(score_row))
            payload["_rank"] = best_wrong_score
            buckets["high_score_wrong"].append(payload)

    raw_counts = {key: len(value) for key, value in buckets.items()}
    for key, rows in buckets.items():
        if key == "high_score_wrong":
            rows.sort(key=lambda row: row.get("_rank", ()), reverse=True)
            for row in rows:
                row.pop("_rank", None)
        else:
            rows.sort(key=lambda row: row["parent_id"])
        if limit >= 0:
            buckets[key] = rows[:limit]
    return {"exp_names": exp_names, "counts": raw_counts, "cases": buckets}


def wanted_rollout_ids(case: JsonDict, exp_names: list[str], mode: str) -> set[str]:
    parent = str(case["parent_id"])
    if mode == "all-rollouts":
        return set()

    rollout_ids: set[int] = set()
    if mode in {"selected", "selected-and-correct"}:
        for name in exp_names:
            rollout = (case.get("by_exp") or {}).get(name, {}).get("rollout_id")
            if rollout != "":
                rollout_ids.add(int(rollout))
    if mode in {"correct", "selected-and-correct"}:
        for name in exp_names:
            for rollout in (case.get("correct_rollouts") or {}).get(name, []):
                rollout_ids.add(int(rollout))
    return {f"{parent}__rollout_{rollout}" for rollout in sorted(rollout_ids)}


def select_flat_rows(flat: pd.DataFrame, cases: list[JsonDict], exp_names: list[str], mode: str) -> pd.DataFrame:
    if not cases:
        return flat.iloc[0:0].copy()
    parent_ids = {str(case["parent_id"]) for case in cases}
    if mode == "all-rollouts":
        selected = flat[flat["parent_id"].isin(parent_ids)].copy()
    else:
        ids: set[str] = set()
        for case in cases:
            ids.update(wanted_rollout_ids(case, exp_names, mode))
        selected = flat[flat["id"].isin(ids)].copy()

    selected["_rollout_sort"] = pd.to_numeric(selected["rollout_id"], errors="coerce")
    selected = selected.sort_values(["parent_id", "_rollout_sort", "id"], kind="stable")
    return selected.drop(columns=["_rollout_sort"])


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flat_to_rollout_raw(rows: pd.DataFrame) -> list[JsonDict]:
    if rows.empty:
        return []
    raw_rows: list[JsonDict] = []
    sortable = rows.copy()
    sortable["_rollout_sort"] = pd.to_numeric(sortable["rollout_id"], errors="coerce")
    sortable = sortable.sort_values(["parent_id", "_rollout_sort", "id"], kind="stable")
    for parent_id, group in sortable.groupby("parent_id", sort=True):
        first = group.iloc[0]
        raw: JsonDict = {
            "id": str(parent_id),
            "source": first.get("source", ""),
            "question": first.get("question", ""),
            "gold": first.get("gold", ""),
        }
        for row in group.to_dict("records"):
            rollout_id = int(row["rollout_id"])
            raw[f"response_{rollout_id}"] = row.get("response", "")
            if "finish_reason" in row:
                raw[f"finish_reason_{rollout_id}"] = row.get("finish_reason")
        raw_rows.append(raw)
    return raw_rows


def write_rollout_dir(
    *,
    out_root: Path,
    out_name: str,
    rows: pd.DataFrame,
    manifest: JsonDict,
    force: bool,
) -> Path:
    out_dir = out_root / rollout_dir_name(out_name)
    parquet_path = out_dir / "rollout_flat.parquet"
    jsonl_path = out_dir / "rollout_flat.jsonl"
    raw_path = out_dir / "rollout_raw.jsonl"
    manifest_path = out_dir / "case_manifest.json"
    if out_dir.exists() and not force and (parquet_path.exists() or jsonl_path.exists() or raw_path.exists()):
        raise SystemExit(f"output already exists: {out_dir} (pass --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(parquet_path, index=False)
    row_dicts = rows.to_dict("records")
    write_jsonl(jsonl_path, row_dicts)
    raw_rows = flat_to_rollout_raw(rows)
    write_jsonl(raw_path, raw_rows)
    manifest_path.write_text(
        json.dumps(
            {
                **manifest,
                "rows": len(row_dicts),
                "raw_rows": len(raw_rows),
                "ids": [str(row["id"]) for row in row_dicts],
                "parent_ids_in_raw": [str(row["id"]) for row in raw_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def default_out_name(exp_names: list[str], mode: str, buckets: list[str]) -> str:
    name = "__".join(name.removeprefix("step_proof_") for name in exp_names)
    bucket_part = "all_buckets" if len(buckets) != 1 else buckets[0]
    return f"cases_{sanitize_name(name)}_{sanitize_name(bucket_part)}_{mode}"


def main() -> None:
    args = parse_args()
    if len(args.experiments) not in {1, 2}:
        raise SystemExit("please provide one or two experiment names")

    exp_names = [normalize_exp_name(name) for name in args.experiments]
    loaded = {name: load_experiment(args.step_proof_root, name) for name in exp_names}
    mined = mine_cases(loaded, args.limit)
    cases_by_bucket: dict[str, list[JsonDict]] = mined["cases"]
    requested_buckets = args.bucket or [key for key, rows in cases_by_bucket.items() if rows]
    unknown = sorted(set(requested_buckets) - set(cases_by_bucket))
    if unknown:
        raise SystemExit(f"unknown bucket(s): {unknown}; available: {sorted(cases_by_bucket)}")

    flat = load_flat(args.source_flat)
    selected_cases = [case for bucket in requested_buckets for case in cases_by_bucket.get(bucket, [])]
    selected_rows = select_flat_rows(flat, selected_cases, exp_names, args.mode)
    if selected_cases and selected_rows.empty:
        raise SystemExit("cases were found, but no matching rollout rows were found in --source-flat")

    out_name = args.out_name.strip() or default_out_name(exp_names, args.mode, requested_buckets)
    manifest = {
        "source_flat": str(args.source_flat),
        "experiments": exp_names,
        "mode": args.mode,
        "limit": args.limit,
        "buckets": requested_buckets,
        "counts": mined["counts"],
        "case_count": len(selected_cases),
        "parent_ids": sorted({str(case["parent_id"]) for case in selected_cases}),
        "cases": {bucket: cases_by_bucket.get(bucket, []) for bucket in requested_buckets},
    }
    out_dir = write_rollout_dir(
        out_root=args.out_root,
        out_name=out_name,
        rows=selected_rows,
        manifest=manifest,
        force=args.force,
    )

    per_bucket_dirs: dict[str, str] = {}
    if args.write_per_bucket:
        base = out_name.removeprefix("rollout_")
        for bucket in requested_buckets:
            bucket_cases = cases_by_bucket.get(bucket, [])
            bucket_rows = select_flat_rows(flat, bucket_cases, exp_names, args.mode)
            bucket_manifest = {
                **manifest,
                "buckets": [bucket],
                "case_count": len(bucket_cases),
                "parent_ids": sorted({str(case["parent_id"]) for case in bucket_cases}),
                "cases": {bucket: bucket_cases},
            }
            bucket_dir = write_rollout_dir(
                out_root=args.out_root,
                out_name=f"{base}_{bucket}",
                rows=bucket_rows,
                manifest=bucket_manifest,
                force=args.force,
            )
            per_bucket_dirs[bucket] = str(bucket_dir)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] experiments: {', '.join(exp_names)}")
    print(f"[done] buckets: {', '.join(requested_buckets) if requested_buckets else '<none>'}")
    print(f"[done] cases: {len(selected_cases)}")
    print(f"[done] rollout rows: {len(selected_rows)}")
    print(f"[done] out -> {out_dir}")
    print()
    print("Use these overrides for the next step-proof experiment:")
    print(f"  stage1.parquet_dir={out_dir}")
    print("  stage1.parquet_glob=rollout_flat.parquet")
    print("  stage1.limit=-1")
    print("  stage2.limit=-1")
    print("  stage3.limit=-1")
    if args.out_root.resolve() == DEFAULT_OUT_ROOT.resolve():
        print(f"  rollout_name={out_dir.name.removeprefix('rollout_')}")
    if per_bucket_dirs:
        print()
        print("Per-bucket outputs:")
        for bucket, path in per_bucket_dirs.items():
            print(f"  {bucket}: {path}")


if __name__ == "__main__":
    main()
