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
class BoolStats:
    n: int = 0
    true_n: int = 0

    def add(self, v: bool) -> None:
        self.n += 1
        if bool(v):
            self.true_n += 1

    def rate(self) -> float:
        return (self.true_n / self.n) if self.n else 0.0


@dataclass
class MetricAgg:
    stats: OnlineStats = field(default_factory=OnlineStats)
    missing: int = 0
    invalid: int = 0


@dataclass
class StepAgg:
    step: int
    metrics: Dict[str, MetricAgg] = field(default_factory=dict)
    bools: Dict[str, BoolStats] = field(default_factory=dict)
    missing_trace: int = 0
    invalid_trace: int = 0

    def metric(self, name: str) -> MetricAgg:
        if name not in self.metrics:
            self.metrics[name] = MetricAgg()
        return self.metrics[name]

    def bool_stat(self, name: str) -> BoolStats:
        if name not in self.bools:
            self.bools[name] = BoolStats()
        return self.bools[name]


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
        return x if math.isfinite(x) else None
    if isinstance(value, str):
        try:
            x = float(value.strip())
        except Exception:
            return None
        return x if math.isfinite(x) else None
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 0/1 style
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 step 统计 rollouts[].fdg_reward_trace 的各项指标分布（流式读取 jsonl）"
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--trace-path",
        default="fdg_reward_trace",
        help="rollouts 内 trace 字段名/路径，默认 fdg_reward_trace（支持 a.b.c 形式）",
    )
    parser.add_argument(
        "--metrics",
        default="score,structure_score,formalizer_score,prover_score,final_answer_score,length_penalty",
        help="需要统计的数值字段（逗号分隔），从 trace 内取",
    )
    parser.add_argument(
        "--bools",
        default="valid_json,validator_passed",
        help="需要统计的布尔字段（逗号分隔），从 trace 内取",
    )
    args = parser.parse_args()

    metric_names = [s.strip() for s in str(args.metrics).split(",") if s.strip()]
    bool_names = [s.strip() for s in str(args.bools).split(",") if s.strip()]
    trace_path = str(args.trace_path).strip()
    trace_parts = [p for p in trace_path.split(".") if p]

    per_step: Dict[int, StepAgg] = {}
    global_metrics: Dict[str, MetricAgg] = {m: MetricAgg() for m in metric_names}
    global_bools: Dict[str, BoolStats] = {b: BoolStats() for b in bool_names}

    totals = {
        "records": 0,
        "rollouts_seen": 0,
        "missing_step_records": 0,
        "missing_rollouts_records": 0,
        "missing_trace_rollouts": 0,
        "invalid_trace_rollouts": 0,
    }

    for rec in iter_jsonl(args.input_jsonl):
        totals["records"] += 1
        step = _to_int_step(rec.get("step"))
        if step is None:
            totals["missing_step_records"] += 1
            continue

        agg = per_step.get(step)
        if agg is None:
            agg = StepAgg(step=step)
            per_step[step] = agg

        rollouts = rec.get("rollouts")
        if not isinstance(rollouts, list):
            totals["missing_rollouts_records"] += 1
            continue

        for ro in rollouts:
            totals["rollouts_seen"] += 1
            if not isinstance(ro, dict):
                agg.invalid_trace += 1
                totals["invalid_trace_rollouts"] += 1
                continue

            trace_val: Any = ro
            for part in trace_parts:
                if not isinstance(trace_val, dict):
                    trace_val = None
                    break
                trace_val = trace_val.get(part)

            if trace_val is None:
                agg.missing_trace += 1
                totals["missing_trace_rollouts"] += 1
                continue
            if not isinstance(trace_val, dict):
                agg.invalid_trace += 1
                totals["invalid_trace_rollouts"] += 1
                continue

            trace = trace_val

            for name in metric_names:
                raw = trace.get(name)
                m = agg.metric(name)
                gm = global_metrics[name]
                if raw is None:
                    m.missing += 1
                    gm.missing += 1
                    continue
                x = _to_float(raw)
                if x is None:
                    m.invalid += 1
                    gm.invalid += 1
                    continue
                m.stats.add(x)
                gm.stats.add(x)

            for name in bool_names:
                raw = trace.get(name)
                bs = agg.bool_stat(name)
                gbs = global_bools[name]
                v = _to_bool(raw)
                if v is None:
                    # treat as missing for bool; keep n unchanged
                    continue
                bs.add(v)
                gbs.add(v)

    steps_sorted = sorted(per_step.keys())
    by_step_rows: List[Dict[str, Any]] = []
    for step in steps_sorted:
        agg = per_step[step]
        row: Dict[str, Any] = {
            "step": step,
            "missing_trace": agg.missing_trace,
            "invalid_trace": agg.invalid_trace,
            "metrics": {},
            "bools": {},
        }
        for name in metric_names:
            m = agg.metric(name)
            row["metrics"][name] = {
                "n": m.stats.n,
                "mean": m.stats.mean if m.stats.n else 0.0,
                "std_pop": m.stats.std_pop() if m.stats.n else 0.0,
                "min": m.stats.min_v if m.stats.n else None,
                "max": m.stats.max_v if m.stats.n else None,
                "missing": m.missing,
                "invalid": m.invalid,
            }
        for name in bool_names:
            b = agg.bool_stat(name)
            row["bools"][name] = {
                "n": b.n,
                "true_n": b.true_n,
                "true_rate": b.rate(),
            }
        by_step_rows.append(row)

    out_payload: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "trace_path": trace_path,
        "metrics": metric_names,
        "bools": bool_names,
        "totals": totals,
        "global": {
            "metrics": {
                name: {
                    "n": global_metrics[name].stats.n,
                    "mean": global_metrics[name].stats.mean if global_metrics[name].stats.n else 0.0,
                    "std_pop": global_metrics[name].stats.std_pop()
                    if global_metrics[name].stats.n
                    else 0.0,
                    "min": global_metrics[name].stats.min_v if global_metrics[name].stats.n else None,
                    "max": global_metrics[name].stats.max_v if global_metrics[name].stats.n else None,
                    "missing": global_metrics[name].missing,
                    "invalid": global_metrics[name].invalid,
                }
                for name in metric_names
            },
            "bools": {
                name: {
                    "n": global_bools[name].n,
                    "true_n": global_bools[name].true_n,
                    "true_rate": global_bools[name].rate(),
                }
                for name in bool_names
            },
        },
        "by_step": by_step_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote json: {args.out_json}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)

            # header: step + trace missing/invalid + each metric summary + each bool rate
            header = ["step", "missing_trace", "invalid_trace"]
            for name in metric_names:
                header += [
                    f"{name}.n",
                    f"{name}.mean",
                    f"{name}.std_pop",
                    f"{name}.min",
                    f"{name}.max",
                    f"{name}.missing",
                    f"{name}.invalid",
                ]
            for name in bool_names:
                header += [f"{name}.n", f"{name}.true_n", f"{name}.true_rate"]
            w.writerow(header)

            for row in by_step_rows:
                out_row: List[Any] = [row["step"], row["missing_trace"], row["invalid_trace"]]
                metrics_obj: Dict[str, Any] = row["metrics"]
                bools_obj: Dict[str, Any] = row["bools"]
                for name in metric_names:
                    m = metrics_obj[name]
                    out_row += [
                        m["n"],
                        m["mean"],
                        m["std_pop"],
                        m["min"],
                        m["max"],
                        m["missing"],
                        m["invalid"],
                    ]
                for name in bool_names:
                    b = bools_obj[name]
                    out_row += [b["n"], b["true_n"], b["true_rate"]]
                w.writerow(out_row)
        print(f"[done] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()

