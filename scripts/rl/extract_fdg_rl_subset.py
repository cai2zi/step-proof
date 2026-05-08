from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from omegaconf import OmegaConf

from proofflow.rl_fdg.dataset import build_rl_examples, write_json_manifest, write_verl_parquet

import sys;
sys.path.append("/root/autodl-tmp/step-proof")
DEFAULT_SOURCE_FILE = Path("/root/autodl-tmp/data_raw/ODA-Math-460k/data_2/shuffled_all.parquet")
DEFAULT_CONFIG_FILE = Path("configs/rl/fdg_grpo.yaml")


def _resolve_from_config(config_path: Path) -> tuple[Path, Path]:
    cfg = OmegaConf.load(str(config_path))
    container = OmegaConf.to_container(cfg, resolve=True)
    train_file = Path(str(container["train_file"]))
    val_file = Path(str(container["val_file"]))
    return train_file, val_file


def _require_columns(df: pd.DataFrame, required: List[str], source_file: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"{source_file} is missing required columns: {missing}")


def _build_rows(
    df: pd.DataFrame,
    *,
    source_file: Path,
    start: int,
    count: int,
    id_column: str,
    question_column: str,
    response_column: str,
    answer_column: str | None,
) -> List[Dict[str, Any]]:
    sliced = df.iloc[start : start + count].copy()
    rows: List[Dict[str, Any]] = []
    for offset, (_, row) in enumerate(sliced.iterrows()):
        rows.append(
            {
                "record_id": str(row[id_column]),
                "problem_text": str(row[question_column]),
                "solution_or_cot": str(row[response_column]),
                "reference_answer": str(row[answer_column]) if answer_column and answer_column in row else "",
                "source_file": str(source_file),
                "source_row_pos": start + offset,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first 10k train rows and the next 100 val rows from shuffled_all.parquet, "
            "then convert them directly into verl RL parquet files."
        )
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Source parquet file, expected to be pre-shuffled.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="RL config file used to resolve train_file and val_file output paths.",
    )
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--fdg-prompt", default="fdg_origin4")
    parser.add_argument("--include-think-in-dag", action="store_true")
    parser.add_argument("--data-source", default="fdg_builder")
    args = parser.parse_args()

    if not args.source_file.is_file():
        raise SystemExit(f"Source parquet file not found: {args.source_file}")
    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    train_out, val_out = _resolve_from_config(args.config)
    df = pd.read_parquet(args.source_file)
    _require_columns(
        df,
        [args.id_column, args.question_column, args.response_column],
        args.source_file,
    )

    total_needed = args.train_count + args.val_count
    if len(df) < total_needed:
        raise SystemExit(
            f"Source parquet has only {len(df)} rows, but {total_needed} rows are required."
        )

    train_rows = _build_rows(
        df,
        source_file=args.source_file,
        start=0,
        count=args.train_count,
        id_column=args.id_column,
        question_column=args.question_column,
        response_column=args.response_column,
        answer_column=args.answer_column,
    )
    val_rows = _build_rows(
        df,
        source_file=args.source_file,
        start=args.train_count,
        count=args.val_count,
        id_column=args.id_column,
        question_column=args.question_column,
        response_column=args.response_column,
        answer_column=args.answer_column,
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

    write_verl_parquet([item.to_verl_record() for item in train_examples], train_out)
    write_verl_parquet([item.to_verl_record() for item in val_examples], val_out)
    write_json_manifest(
        {
            "source_file": str(args.source_file),
            "config_file": str(args.config),
            "fdg_prompt": args.fdg_prompt,
            "include_think_in_dag": bool(args.include_think_in_dag),
            "data_source": args.data_source,
            "train_count": len(train_examples),
            "val_count": len(val_examples),
            "train_range": [0, args.train_count],
            "val_range": [args.train_count, args.train_count + args.val_count],
            "train_output": str(train_out),
            "val_output": str(val_out),
        },
        train_out.parent / "manifests" / "subset_manifest.json",
    )

    print(f"Train RL parquet written: {len(train_examples)} -> {train_out}")
    print(f"Val RL parquet written: {len(val_examples)} -> {val_out}")
    print(
        f"Source slice summary: train=[0:{args.train_count}), "
        f"val=[{args.train_count}:{args.train_count + args.val_count})"
    )
    print(
        f"Prompt settings: fdg_prompt={args.fdg_prompt}, "
        f"include_think_in_dag={bool(args.include_think_in_dag)}"
    )


if __name__ == "__main__":
    main()
