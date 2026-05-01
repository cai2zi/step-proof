"""Stage 3 entrypoint: read Stage 2 FDG JSONL and batch-run proving."""
from __future__ import annotations

import asyncio
from pathlib import Path

from proofflow.fdg_stage3_runner import FDGStage3Runner, build_arg_parser
from proofflow.graph_mode import ensure_fdg_jsonl


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.infile.is_file():
        raise SystemExit(f"--infile not found: {args.infile}")
    if not Path(args.mathlib_path).is_dir():
        raise SystemExit(f"--mathlib-path is not a directory: {args.mathlib_path}")
    ensure_fdg_jsonl(args.infile)
    runner = FDGStage3Runner(args)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
