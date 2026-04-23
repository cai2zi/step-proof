from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


JsonDict = Dict[str, Any]


def _load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _node_fully_verified(node: JsonDict) -> bool:
    role = str(node.get("role", "")).strip()
    formalization = node.get("formalization") or {}
    solved_lemma = node.get("solved_lemma") or {}

    # Match vis.py semantics:
    # - condition/context: green when formalization.lean_pass is True
    # - claim/final: green when solved_lemma.lean_verify is True
    if role in {"condition", "context"}:
        return bool(formalization.get("lean_pass", False))
    return bool(solved_lemma.get("lean_verify", False))


def _record_all_nodes_fully_verified(rec: JsonDict) -> bool:
    nodes = rec.get("results", {}).get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return False
    return all(_node_fully_verified(node) for node in nodes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count Stage 3 samples where all nodes are Fully verified.",
    )
    parser.add_argument(
        "--stage3-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "result_stage3" / "stage3_results.jsonl",
        help="Path to stage3 results JSONL",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print record_id list that satisfy all-nodes-fully-verified",
    )
    args = parser.parse_args()

    if not args.stage3_jsonl.is_file():
        raise SystemExit(f"stage3 JSONL not found: {args.stage3_jsonl}")

    rows = _load_jsonl(args.stage3_jsonl)
    total = len(rows)

    passed_ids: List[str] = []
    for rec in rows:
        rid = str(rec.get("meta", {}).get("record_id", "")).strip()
        if _record_all_nodes_fully_verified(rec):
            passed_ids.append(rid)

    passed = len(passed_ids)
    ratio = (passed / total * 100.0) if total else 0.0

    print(f"total_samples: {total}")
    print(f"all_nodes_fully_verified: {passed}")
    print(f"ratio: {ratio:.2f}%")

    if args.show_ids:
        print("record_ids:")
        for rid in passed_ids:
            print(rid)


if __name__ == "__main__":
    main()
