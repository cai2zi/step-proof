#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import (
    load_config,
    read_jsonl,
    split_rollout_id,
    step_proof_dir,
    step_proof_results_dir,
    step_proof_rollout_dir,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rollout candidates by stage3 prove success on non-root facts.",
    )
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def _facts_from_stage3_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = record.get("results") or {}
    facts = results.get("facts")
    return facts if isinstance(facts, list) else []


def _prove_required_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [fact for fact in facts if fact.get("parent_fact_ids")]


def _score_facts(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    prove_facts = _prove_required_facts(facts)
    prove_required_nodes = len(prove_facts)
    prove_success_nodes = 0
    lean_verify_nodes = 0
    lean_pass_nodes = 0
    for fact in prove_facts:
        solved = fact.get("solved_lemma") or {}
        if fact.get("prove_status") == "success":
            prove_success_nodes += 1
        if solved.get("lean_verify") is True:
            lean_verify_nodes += 1
        if solved.get("lean_pass") is True:
            lean_pass_nodes += 1
    return {
        "prove_required_nodes": prove_required_nodes,
        "prove_success_nodes": prove_success_nodes,
        "lean_verify_nodes": lean_verify_nodes,
        "lean_pass_nodes": lean_pass_nodes,
        "success_ratio": (
            prove_success_nodes / prove_required_nodes if prove_required_nodes else 0.0
        ),
    }


def _sort_key(row: Dict[str, Any]) -> tuple:
    return (
        -float(row["success_ratio"]),
        -int(row["prove_success_nodes"]),
        -int(row["lean_pass_nodes"]),
        int(row["rollout_id"]),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    flat_path = step_proof_rollout_dir(cfg) / "rollout_flat.parquet"
    stage3_path = step_proof_results_dir(cfg) / "result_stage3" / "stage3_results.jsonl"
    out_dir = step_proof_dir(cfg)
    scores_path = out_dir / "scores.jsonl"
    selected_path = out_dir / "selected_step_proof.jsonl"

    flat = pd.read_parquet(flat_path)
    candidate_by_id = {str(row["id"]): row for row in flat.to_dict("records")}
    scores: Dict[str, Dict[str, Any]] = {}

    if stage3_path.is_file():
        for rec in read_jsonl(stage3_path):
            rid = str((rec.get("meta") or {}).get("record_id") or "")
            if not rid:
                continue
            scores[rid] = _score_facts(_facts_from_stage3_record(rec))
    else:
        print(f"[warn] missing stage3 results: {stage3_path}")

    rows = []
    for rid, candidate in candidate_by_id.items():
        parent_id, rollout_id = split_rollout_id(rid)
        score = scores.get(
            rid,
            {
                "prove_required_nodes": 0,
                "prove_success_nodes": -1,
                "lean_verify_nodes": -1,
                "lean_pass_nodes": -1,
                "success_ratio": 0.0,
            },
        )
        rows.append(
            {
                "id": rid,
                "parent_id": parent_id,
                "rollout_id": rollout_id,
                "source": candidate.get("source", ""),
                "question": candidate.get("question", ""),
                "response": candidate.get("response", ""),
                "gold": candidate.get("gold", ""),
                **score,
            }
        )

    rows.sort(key=lambda row: (row["parent_id"], _sort_key(row)))
    write_jsonl(scores_path, rows)

    selected = []
    for parent_id, group in pd.DataFrame(rows).groupby("parent_id", sort=True):
        best = sorted(group.to_dict("records"), key=_sort_key)[0]
        selected.append(
            {
                "id": parent_id,
                "source": best["source"],
                "selected_rollout_id": int(best["rollout_id"]),
                "selected_response": best["response"],
                "gold": best["gold"],
                "score": {
                    key: best[key]
                    for key in (
                        "prove_required_nodes",
                        "prove_success_nodes",
                        "lean_verify_nodes",
                        "lean_pass_nodes",
                        "success_ratio",
                    )
                },
            }
        )
    write_jsonl(selected_path, selected)
    print(f"[done] scores -> {scores_path}")
    print(f"[done] selected -> {selected_path}")
    print(f"[done] selected {len(selected)} parent problem(s)")


if __name__ == "__main__":
    main()
