#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    has_unclosed_think_block,
    load_config,
    read_jsonl,
    rollout_dir,
    rollout_record_id,
    rollout_response_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten rollout_raw.jsonl into step-proof input.")
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    n = int(cfg["rollout"]["n"])
    out_dir = rollout_dir(cfg)
    rollout_path = out_dir / "rollout_raw.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rollout_flat.jsonl"
    parquet_path = out_dir / "rollout_flat.parquet"
    skipped_path = out_dir / "rollout_flat_skipped_unclosed_think.jsonl"

    records = []
    skipped = []
    for row in read_jsonl(rollout_path):
        parent_id = str(row["id"])
        for rollout_id in range(1, n + 1):
            response = row.get(rollout_response_key(rollout_id))
            if response is None or not str(response).strip():
                continue
            record_id = rollout_record_id(parent_id, rollout_id)
            if has_unclosed_think_block(response):
                skipped.append(
                    {
                        "id": record_id,
                        "parent_id": parent_id,
                        "rollout_id": rollout_id,
                        "source": row.get("source", ""),
                        "reason": "unclosed_think_block",
                    }
                )
                continue
            records.append(
                {
                    "id": record_id,
                    "parent_id": parent_id,
                    "rollout_id": rollout_id,
                    "source": row.get("source", ""),
                    "question": row.get("question", ""),
                    "response": response,
                    "gold": row.get("gold", ""),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit(f"no rollout responses found in {rollout_path}")
    df.to_parquet(parquet_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with skipped_path.open("w", encoding="utf-8") as f:
        for rec in skipped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] flattened {len(df)} rollout response(s)")
    print(f"[done] skipped unclosed <think> response(s): {len(skipped)}")
    print(f"[done] jsonl -> {jsonl_path}")
    print(f"[done] parquet -> {parquet_path}")
    print(f"[done] skipped -> {skipped_path}")


if __name__ == "__main__":
    main()
