from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


DEFAULT_SOURCE_FILE = Path("/root/autodl-tmp/data_raw/ODA-Math-460k/data_2/shuffled_all.parquet")
DEFAULT_CONFIG_FILE = Path("configs/rl/fdg_grpo.yaml")


def _resolve_from_config(config_path: Path) -> tuple[Path, Path]:
    cfg = OmegaConf.load(str(config_path))
    container = OmegaConf.to_container(cfg, resolve=True)
    train_file = Path(str(container["train_file"]))
    val_file = Path(str(container["val_file"]))
    return train_file, val_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the first 10k train rows and the next 100 val rows from shuffled_all.parquet."
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
    parser.add_argument("--val-count", type=int, default=100)
    args = parser.parse_args()

    if not args.source_file.is_file():
        raise SystemExit(f"Source parquet file not found: {args.source_file}")
    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    train_out, val_out = _resolve_from_config(args.config)
    df = pd.read_parquet(args.source_file)
    total_needed = args.train_count + args.val_count
    if len(df) < total_needed:
        raise SystemExit(
            f"Source parquet has only {len(df)} rows, but {total_needed} rows are required."
        )

    train_df = df.iloc[: args.train_count].copy()
    val_df = df.iloc[args.train_count : args.train_count + args.val_count].copy()

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"Train rows written: {len(train_df)} -> {train_out}")
    print(f"Val rows written: {len(val_df)} -> {val_out}")
    print(
        f"Source slice summary: train=[0:{args.train_count}), "
        f"val=[{args.train_count}:{args.train_count + args.val_count})"
    )


if __name__ == "__main__":
    main()
