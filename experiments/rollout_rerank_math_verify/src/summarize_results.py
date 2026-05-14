#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from common import load_config, math_verify_dir, summary_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize random, pass@1, step-proof, and pass@k Math-Verify results.",
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _accuracy(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return math.nan
    return sum(int(row.get("is_correct", "0") or 0) for row in rows) / len(rows)


def _per_source(rows: List[Dict[str, str]]) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("source", ""), []).append(row)
    return {source: _accuracy(group) for source, group in sorted(groups.items())}


def _per_source_pass_at_k(parent_ok: Dict[str, bool], parent_source: Dict[str, str]) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    for pid, ok in parent_ok.items():
        source = parent_source.get(pid, "")
        totals[source] = totals.get(source, 0) + 1
        corrects[source] = corrects.get(source, 0) + (1 if ok else 0)
    return {source: (corrects.get(source, 0) / total if total else math.nan) for source, total in sorted(totals.items())}


def _is_rollout_one(row: Dict[str, str]) -> bool:
    """pass@1: use only the first rollout (index 1) per parent."""
    rid = row.get("rollout_id")
    if rid is not None and str(rid).strip() != "":
        try:
            return int(rid) == 1
        except (TypeError, ValueError):
            pass
    lid = row.get("id", "")
    if "__rollout_" in lid:
        try:
            return int(lid.rsplit("__rollout_", 1)[-1]) == 1
        except ValueError:
            pass
    return False


def _rollout_id(row: Dict[str, str]) -> int | None:
    rid = row.get("rollout_id")
    if rid is not None and str(rid).strip() != "":
        try:
            return int(rid)
        except (TypeError, ValueError):
            pass
    lid = row.get("id", "")
    if "__rollout_" in lid:
        try:
            return int(lid.rsplit("__rollout_", 1)[-1])
        except ValueError:
            pass
    return None


def _write_metrics_csv(
    *,
    out_path: Path,
    metrics: Dict[str, Any],
    pass_at_k_key: str,
) -> None:
    """Write a wide CSV: columns are sources + avg, rows are metrics."""
    random_payload = metrics.get("random") or {}
    runs = random_payload.get("runs") or []

    def _sources_from_per_source(per_source: Any) -> List[str]:
        if not isinstance(per_source, dict):
            return []
        return [str(k) for k in per_source.keys() if str(k)]

    sources: List[str] = []
    # collect from random runs
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                sources.extend(_sources_from_per_source(run.get("per_source")))
    # collect from deterministic metrics
    for name in ("pass_at_1", pass_at_k_key, "step_proof_best"):
        payload = metrics.get(name)
        if isinstance(payload, dict):
            sources.extend(_sources_from_per_source(payload.get("per_source")))
    sources = sorted(set(sources))

    def _row(name: str, per_source: Dict[str, Any], avg: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {"metric": name}
        for source in sources:
            row[source] = per_source.get(source, "")
        row["avg"] = avg
        return row

    rows: List[Dict[str, Any]] = []
    # first 3 rows: random_seed{}
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            seed = run.get("seed", "")
            rows.append(
                _row(
                    f"random_seed_{seed}",
                    per_source=run.get("per_source") or {},
                    avg=run.get("accuracy", ""),
                )
            )

    # pass@1
    pass_at_1 = metrics.get("pass_at_1") or {}
    if isinstance(pass_at_1, dict):
        rows.append(
            _row(
                "pass@1",
                per_source=pass_at_1.get("per_source") or {},
                avg=pass_at_1.get("accuracy", ""),
            )
        )

    # pass@k
    pass_at_k = metrics.get(pass_at_k_key) or {}
    if isinstance(pass_at_k, dict):
        rows.append(
            _row(
                f"pass@{pass_at_k_key.rsplit('_', 1)[-1]}",
                per_source=pass_at_k.get("per_source") or {},
                avg=pass_at_k.get("accuracy", ""),
            )
        )

    # ours
    ours = metrics.get("step_proof_best") or {}
    if isinstance(ours, dict):
        rows.append(
            _row(
                "ours",
                per_source=ours.get("per_source") or {},
                avg=ours.get("accuracy", ""),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["metric", *sources, "avg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mv_dir = math_verify_dir(cfg)
    out_summary_dir = summary_dir(cfg)
    out_summary_dir.mkdir(parents=True, exist_ok=True)

    random_metrics = []
    for seed in cfg["math_verify"].get("random_seeds", [0, 1, 2]):
        path = mv_dir / f"random_seed_{seed}_eval.jsonl"
        rows = _read_jsonl(path)
        random_metrics.append(
            {
                "seed": int(seed),
                "accuracy": _accuracy(rows),
                "per_source": _per_source(rows),
            }
        )

    step_rows = _read_jsonl(mv_dir / "step_proof_best_eval.jsonl")
    all_rows = _read_jsonl(mv_dir / "all_rollouts_eval.jsonl")
    pass_at_k_by_parent: Dict[str, bool] = {}
    parent_source: Dict[str, str] = {}
    for row in all_rows:
        parent_id = row.get("parent_id") or row.get("id", "").rsplit("__rollout_", 1)[0]
        pass_at_k_by_parent[parent_id] = pass_at_k_by_parent.get(parent_id, False) or row.get("is_correct") == "1"
        if parent_id not in parent_source or not parent_source[parent_id]:
            parent_source[parent_id] = row.get("source", "") or ""

    step_by_parent = {row.get("parent_id") or row["id"]: row for row in step_rows}
    rollout_ids = [_rollout_id(row) for row in all_rows]
    k = max([rid for rid in rollout_ids if rid is not None], default=0)
    pass_at_k_total = len(pass_at_k_by_parent)
    pass_at_k_correct = sum(1 for value in pass_at_k_by_parent.values() if value)
    pass_at_k_accuracy = pass_at_k_correct / pass_at_k_total if pass_at_k_total else math.nan
    pass_at_k_per_source = _per_source_pass_at_k(pass_at_k_by_parent, parent_source)
    pass_at_1_rows = [row for row in all_rows if _is_rollout_one(row)]
    possible = [pid for pid, ok in pass_at_k_by_parent.items() if ok]
    selected_hits = sum(1 for pid in possible if step_by_parent.get(pid, {}).get("is_correct") == "1")
    selection_hit_rate = selected_hits / len(possible) if possible else math.nan

    random_accs = [item["accuracy"] for item in random_metrics]
    pass_at_k_key = f"pass_at_{k}" if k > 0 else "pass_at_k"
    metrics = {
        "total": len(step_rows),
        "random": {
            "runs": random_metrics,
            "mean_accuracy": mean(random_accs) if random_accs else math.nan,
            "std_accuracy": pstdev(random_accs) if len(random_accs) > 1 else 0.0,
        },
        "pass_at_1": {
            "accuracy": _accuracy(pass_at_1_rows),
            "per_source": _per_source(pass_at_1_rows),
            "total": len(pass_at_1_rows),
        },
        "step_proof_best": {
            "accuracy": _accuracy(step_rows),
            "per_source": _per_source(step_rows),
        },
        pass_at_k_key: {
            "accuracy": pass_at_k_accuracy,
            "per_source": pass_at_k_per_source,
            "correct": pass_at_k_correct,
            "total": pass_at_k_total,
        },
        "selection": {
            f"hit_rate_on_{pass_at_k_key}_correct": selection_hit_rate,
            "hits": selected_hits,
            f"{pass_at_k_key}_correct_total": len(possible),
        },
    }

    metrics_path = out_summary_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_metrics_csv(
        out_path=out_summary_dir / "metrics.csv",
        metrics=metrics,
        pass_at_k_key=pass_at_k_key,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
