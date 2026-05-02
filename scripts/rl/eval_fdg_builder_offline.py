from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from proofflow.rl_fdg.evaluator import FDGRLEvaluator, load_evaluator_config
from proofflow.rl_fdg.reward_types import CandidateGraphInput


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for FDG builder outputs.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, default=Path("configs/rl/fdg_reward.yaml"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    evaluator = FDGRLEvaluator(load_evaluator_config(args.reward_config))
    outputs: List[Dict[str, Any]] = []
    total_score = 0.0
    total = 0

    try:
        for start in range(0, len(rows), args.batch_size):
            batch_inputs = [
                CandidateGraphInput(
                    record_id=str(row.get("record_id", "")),
                    problem_text=str(row.get("problem_text", "")),
                    solution_or_cot=str(row.get("solution_or_cot", "")),
                    model_output=str(row.get("model_output", "")),
                    ground_truth=str(row.get("ground_truth", "")),
                    data_source=str(row.get("data_source", "fdg_builder")),
                    extra_info=dict(row.get("extra_info") or {}),
                )
                for row in rows[start : start + args.batch_size]
            ]
            batch_outputs = evaluator.evaluate_batch_sync(batch_inputs)
            for item in batch_outputs:
                outputs.append(item.to_dict())
                total_score += item.score
                total += 1
    finally:
        evaluator.close()

    dump_jsonl(args.out_jsonl, outputs)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "num_examples": total,
                    "avg_score": (total_score / total) if total else 0.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
