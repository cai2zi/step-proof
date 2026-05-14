#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import load_config, math_verify_dir, read_jsonl, step_proof_dir, step_proof_rollout_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Math-Verify CSV inputs for random and step-proof selections.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rollout_responses(rec: Dict[str, Any]) -> List[tuple[int, Any]]:
    responses = []
    for key, value in rec.items():
        if not key.startswith("response_"):
            continue
        suffix = key.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        responses.append((int(suffix), value))
    return sorted(responses)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = math_verify_dir(cfg)
    rollout_raw = list(read_jsonl(step_proof_rollout_dir(cfg) / "rollout_raw.jsonl"))
    selected = list(read_jsonl(step_proof_dir(cfg) / "selected_step_proof.jsonl"))

    all_rollout_rows: List[Dict[str, Any]] = []
    for rec in rollout_raw:
        for rollout_id, response in _rollout_responses(rec):
            if response is None or not str(response).strip():
                continue
            all_rollout_rows.append(
                {
                    "id": f"{rec['id']}__rollout_{rollout_id}",
                    "parent_id": rec["id"],
                    "source": rec.get("source", ""),
                    "rollout_id": rollout_id,
                    "answer": response,
                    "gold": rec.get("gold", ""),
                }
            )
    _write_jsonl(out_dir / "all_rollouts.jsonl", all_rollout_rows)

    by_parent = {}
    for row in all_rollout_rows:
        by_parent.setdefault(row["parent_id"], []).append(row)

    for seed in cfg["math_verify"].get("random_seeds", [0, 1, 2]):
        rng = random.Random(int(seed))
        rows = []
        for parent_id in sorted(by_parent):
            rows.append(dict(rng.choice(by_parent[parent_id])))
        _write_jsonl(out_dir / f"random_seed_{seed}.jsonl", rows)

    step_rows = []
    for rec in selected:
        score = rec.get("score") or {}
        step_rows.append(
            {
                "id": rec["id"],
                "parent_id": rec["id"],
                "source": rec.get("source", ""),
                "rollout_id": rec.get("selected_rollout_id", ""),
                "answer": rec.get("selected_response", ""),
                "gold": rec.get("gold", ""),
                **{f"score_{key}": value for key, value in score.items()},
            }
        )
    _write_jsonl(out_dir / "step_proof_best.jsonl", step_rows)
    print(f"[done] Math-Verify inputs -> {out_dir}")


if __name__ == "__main__":
    main()
