"""
Stage 3 entrypoint: read Stage 2 graph-form JSONL and batch-run prove.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from proofflow.stage3_runner import Stage3Runner, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.infile.is_file():
        raise SystemExit(f"--infile not found: {args.infile}")
    if not Path(args.mathlib_path).is_dir():
        raise SystemExit(f"--mathlib-path is not a directory: {args.mathlib_path}")
    asyncio.run(Stage3Runner(args).run())


if __name__ == "__main__":
    main()
