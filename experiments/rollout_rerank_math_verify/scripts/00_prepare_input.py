#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import exp_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare bench subset for rollout rerank experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--limit-total", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_dir = exp_dir(cfg) / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = Path(data_cfg["bench_path"])
    if bench_path.suffix == ".parquet":
        df = pd.read_parquet(bench_path)
    elif bench_path.suffix == ".jsonl":
        df = pd.read_json(bench_path, lines=True)
    else:
        raise SystemExit(f"unsupported bench file: {bench_path}")

    columns = {
        "id": data_cfg.get("id_column", "id"),
        "source": data_cfg.get("source_column", "source"),
        "question": data_cfg.get("question_column", "question"),
        "gold": data_cfg.get("gold_column", "gold"),
    }
    missing = [col for col in columns.values() if col not in df.columns]
    if missing:
        raise SystemExit(f"missing required column(s): {missing}")

    df = df.rename(columns={v: k for k, v in columns.items()})
    df = df[["id", "source", "question", "gold"]].copy()
    df = df.dropna(subset=["id", "source", "question", "gold"])
    df["id"] = df["id"].astype(str)
    df["source"] = df["source"].astype(str)
    df["question"] = df["question"].astype(str)
    df["gold"] = df["gold"].astype(str)

    sources = data_cfg.get("sources")
    if sources:
        available_sources = set(df["source"].unique().tolist())
        requested_sources = set(str(source) for source in sources)
        unknown_sources = sorted(requested_sources - available_sources)
        if unknown_sources:
            available = ", ".join(sorted(available_sources))
            raise SystemExit(
                f"unknown data.sources value(s): {unknown_sources}. "
                f"Available sources: {available}"
            )
        df = df[df["source"].isin(sources)]
        if df.empty:
            raise SystemExit(f"data.sources selected no records: {sources}")

    seed = int(data_cfg.get("seed", 42))
    limit_per_source = (
        args.limit_per_source
        if args.limit_per_source is not None
        else data_cfg.get("limit_per_source")
    )
    if limit_per_source:
        parts = []
        for _, group in df.groupby("source", sort=True):
            parts.append(
                group.sample(
                    n=min(int(limit_per_source), len(group)),
                    random_state=seed,
                )
            )
        df = pd.concat(parts, ignore_index=True)

    if args.limit_total:
        df = df.sample(n=min(args.limit_total, len(df)), random_state=seed)

    df = df.sort_values(["source", "id"]).reset_index(drop=True)
    parquet_path = out_dir / "bench.parquet"
    jsonl_path = out_dir / "bench.jsonl"
    manifest_path = out_dir / "manifest.json"

    df.to_parquet(parquet_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "total": int(len(df)),
        "sources": {str(k): int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
