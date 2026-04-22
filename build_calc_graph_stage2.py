"""
阶段二入口：读取 stage1 的 graph-v1 JSONL，批量执行 form + prove。
实际调度、状态管理和本地 vLLM/Lean 并发逻辑已拆分到 proofflow/ 下。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from proofflow.stage2_runner import Stage2Runner, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.infile.is_file():
        raise SystemExit(f"--infile not found: {args.infile}")
    if not Path(args.mathlib_path).is_dir():
        raise SystemExit(f"--mathlib-path is not a directory: {args.mathlib_path}")
    asyncio.run(Stage2Runner(args).run())


if __name__ == "__main__":
    main()
