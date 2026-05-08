#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = math.inf
    max_v: float = -math.inf

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x

    def var_pop(self) -> float:
        return self.m2 / self.n if self.n else 0.0

    def std_pop(self) -> float:
        return math.sqrt(self.var_pop())


@dataclass
class StepAgg:
    step: int
    stats: OnlineStats = field(default_factory=OnlineStats)
    missing_reward: int = 0
    invalid_reward: int = 0
    hist: List[int] = field(default_factory=list)

    def ensure_hist(self, n_bins: int) -> None:
        if not self.hist:
            self.hist = [0] * n_bins


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse json at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected object at line {line_no}, got {type(obj).__name__}")
            yield obj


def _to_int_step(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isfinite(x):
            return x
        return None
    if isinstance(value, str):
        try:
            x = float(value.strip())
        except Exception:
            return None
        return x if math.isfinite(x) else None
    return None


def _bin_index(x: float, *, bmin: float, bmax: float, bw: float, n_bins: int) -> Optional[int]:
    if x < bmin or x > bmax:
        return None
    if x == bmax:
        return n_bins - 1
    idx = int((x - bmin) / bw)
    if 0 <= idx < n_bins:
        return idx
    return None


def build_bins(bin_min: float, bin_max: float, bin_width: float) -> Tuple[List[float], List[str]]:
    if not (bin_width > 0):
        raise ValueError("--bin-width must be > 0")
    if not (bin_max > bin_min):
        raise ValueError("--bin-max must be > --bin-min")
    n_bins = int(math.ceil((bin_max - bin_min) / bin_width))
    edges = [bin_min + i * bin_width for i in range(n_bins + 1)]
    labels = [f"[{edges[i]:.4g},{edges[i+1]:.4g})" for i in range(n_bins)]
    labels[-1] = f"[{edges[-2]:.4g},{edges[-1]:.4g}]"  # inclusive last bin
    return edges, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="统计各个 step 的 reward 分布（流式读取 jsonl）")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default="/root/autodl-tmp/step-proof/results/fdg_builder_grpo/reward_samples/train_reward_samples.jsonl",
        help="输入 jsonl，例如 /root/autodl-tmp/step-proof/results/fdg_builder_grpo/reward_samples/train_reward_samples.jsonl",
    )
    parser.add_argument("--out-json", type=Path, default="/root/autodl-tmp/step-proof/results/fdg_builder_grpo/reward_samples/reward_dist_by_step.json", help="输出汇总 json")
    parser.add_argument("--out-csv", type=Path, default="/root/autodl-tmp/step-proof/results/fdg_builder_grpo/reward_samples/reward_dist_by_step.csv", help="输出汇总 csv（可选）")
    parser.add_argument("--bin-min", type=float, default=-1.0, help="直方图最小值（含）")
    parser.add_argument("--bin-max", type=float, default=1.0, help="直方图最大值（含）")
    parser.add_argument("--bin-width", type=float, default=0.05, help="直方图 bin 宽度")
    parser.add_argument(
        "--reward-field",
        default="reward",
        help="rollouts 内 reward 字段名，默认 reward（也可设为 score）",
    )
    args = parser.parse_args()

    edges, bin_labels = build_bins(args.bin_min, args.bin_max, args.bin_width)
    n_bins = len(bin_labels)

    per_step: Dict[int, StepAgg] = {}
    global_stats = OnlineStats()
    global_hist = [0] * n_bins
    total_records = 0
    total_rollouts = 0
    total_missing_step = 0
    total_missing_rollouts = 0

    for rec in iter_jsonl(args.input_jsonl):
        total_records += 1
        step = _to_int_step(rec.get("step"))
        if step is None:
            total_missing_step += 1
            continue
        agg = per_step.get(step)
        if agg is None:
            agg = StepAgg(step=step)
            agg.ensure_hist(n_bins)
            per_step[step] = agg
        elif not agg.hist:
            agg.ensure_hist(n_bins)

        rollouts = rec.get("rollouts")
        if not isinstance(rollouts, list):
            total_missing_rollouts += 1
            continue

        for ro in rollouts:
            total_rollouts += 1
            if not isinstance(ro, dict):
                agg.invalid_reward += 1
                continue
            raw = ro.get(args.reward_field)
            if raw is None:
                agg.missing_reward += 1
                continue
            x = _to_float(raw)
            if x is None:
                agg.invalid_reward += 1
                continue

            agg.stats.add(x)
            global_stats.add(x)
            idx = _bin_index(x, bmin=args.bin_min, bmax=args.bin_max, bw=args.bin_width, n_bins=n_bins)
            if idx is not None:
                agg.hist[idx] += 1
                global_hist[idx] += 1

    steps_sorted = sorted(per_step.keys())
    rows: List[Dict[str, Any]] = []
    for step in steps_sorted:
        agg = per_step[step]
        rows.append(
            {
                "step": step,
                "n": agg.stats.n,
                "mean": agg.stats.mean if agg.stats.n else 0.0,
                "std_pop": agg.stats.std_pop() if agg.stats.n else 0.0,
                "min": agg.stats.min_v if agg.stats.n else None,
                "max": agg.stats.max_v if agg.stats.n else None,
                "missing_reward": agg.missing_reward,
                "invalid_reward": agg.invalid_reward,
                "hist": agg.hist,
            }
        )

    out_payload = {
        "input_jsonl": str(args.input_jsonl),
        "reward_field": args.reward_field,
        "bins": {
            "bin_min": args.bin_min,
            "bin_max": args.bin_max,
            "bin_width": args.bin_width,
            "labels": bin_labels,
            "edges": edges,
        },
        "totals": {
            "records": total_records,
            "rollouts_seen": total_rollouts,
            "missing_step_records": total_missing_step,
            "missing_rollouts_records": total_missing_rollouts,
            "global_reward_n": global_stats.n,
            "global_reward_mean": global_stats.mean if global_stats.n else 0.0,
            "global_reward_std_pop": global_stats.std_pop() if global_stats.n else 0.0,
            "global_reward_min": global_stats.min_v if global_stats.n else None,
            "global_reward_max": global_stats.max_v if global_stats.n else None,
            "global_reward_hist": global_hist,
        },
        "by_step": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote json: {args.out_json}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "step",
                    "n",
                    "mean",
                    "std_pop",
                    "min",
                    "max",
                    "missing_reward",
                    "invalid_reward",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r["step"],
                        r["n"],
                        r["mean"],
                        r["std_pop"],
                        r["min"],
                        r["max"],
                        r["missing_reward"],
                        r["invalid_reward"],
                    ]
                )
        print(f"[done] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()

