from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from proofflow.rl_fdg.dataset import (
    build_rl_examples,
    iter_parquet_examples,
    write_json_manifest,
    write_verl_parquet,
)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total >= 3:
        train_count = max(1, int(total * train_ratio))
        val_count = max(1, int(total * val_ratio))
        if train_count + val_count >= total:
            overflow = train_count + val_count - (total - 1)
            train_count = max(1, train_count - overflow)
        test_count = total - train_count - val_count
        return train_count, val_count, test_count
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = max(0, total - train_count - val_count)
    return train_count, val_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FDG builder RL parquet files for verl.")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.parquet")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fdg-prompt", default="fdg_origin4")
    parser.add_argument("--include-think-in-dag", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--data-source", default="fdg_builder")
    args = parser.parse_args()

    rows = list(
        iter_parquet_examples(
            parquet_dir=args.parquet_dir,
            glob_pattern=args.glob,
            id_column=args.id_column,
            question_column=args.question_column,
            response_column=args.response_column,
            answer_column=args.answer_column,
            limit=args.limit,
        )
    )
    if not rows:
        raise SystemExit("No input rows were found.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    train_count, val_count, _test_count = split_counts(len(rows), args.train_ratio, args.val_ratio)
    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    test_rows = rows[train_count + val_count :]
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit(
            "Prepared split would be empty. Increase --limit or adjust --train-ratio/--val-ratio."
        )

    train_examples = build_rl_examples(
        train_rows,
        prompt_name=args.fdg_prompt,
        include_think_in_dag=args.include_think_in_dag,
        data_source=args.data_source,
        split="train",
    )
    val_examples = build_rl_examples(
        val_rows,
        prompt_name=args.fdg_prompt,
        include_think_in_dag=args.include_think_in_dag,
        data_source=args.data_source,
        split="val",
    )
    test_examples = build_rl_examples(
        test_rows,
        prompt_name=args.fdg_prompt,
        include_think_in_dag=args.include_think_in_dag,
        data_source=args.data_source,
        split="test",
    )

    write_verl_parquet([item.to_verl_record() for item in train_examples], args.out_dir / "train.parquet")
    write_verl_parquet([item.to_verl_record() for item in val_examples], args.out_dir / "val.parquet")
    write_verl_parquet([item.to_verl_record() for item in test_examples], args.out_dir / "test.parquet")
    write_json_manifest(
        {
            "num_total": len(rows),
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "num_test": len(test_examples),
            "fdg_prompt": args.fdg_prompt,
            "include_think_in_dag": bool(args.include_think_in_dag),
            "source_dir": str(args.parquet_dir),
            "glob": args.glob,
        },
        args.out_dir / "manifests" / "manifest.json",
    )


if __name__ == "__main__":
    main()
