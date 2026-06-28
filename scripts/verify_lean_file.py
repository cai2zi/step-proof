#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


STEP_PROOF_ROOT = Path(__file__).resolve().parents[1]
if str(STEP_PROOF_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_PROOF_ROOT))

from proofflow.lean_check import LeanServer  # noqa: E402


def default_mathlib_path() -> Path:
    root = os.environ.get("CZX_ROOT")
    if root:
        return Path(root) / "mathlib4"
    return STEP_PROOF_ROOT.parent / "mathlib4"


def default_temp_root() -> Path:
    root = os.environ.get("CZX_ROOT")
    if root:
        return Path(root) / "czx_work" / "TEMP" / "lean_jobs" / "single_file_verifier"
    return STEP_PROOF_ROOT.parent / "czx_work" / "TEMP" / "lean_jobs" / "single_file_verifier"


def resolve_lean_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.suffix.lower() != ".lean":
        raise ValueError(f"input must be a pure .lean file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Lean file not found: {path}")
    return path.resolve()


def compact_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        text = output.strip()
        return "" if text == "[]" else output
    if isinstance(output, (list, tuple, dict)) and not output:
        return ""
    return json.dumps(output, ensure_ascii=False, indent=2)


async def verify_file(
    server: LeanServer,
    path: Path,
    *,
    add_imports: bool,
    allow_sorry: bool,
) -> bool:
    lean_code = path.read_text(encoding="utf-8")
    started = time.perf_counter()
    lean_pass, lean_verify, output = await server.check_lean_string_async(
        lean_code,
        add_imports=add_imports,
    )
    elapsed = time.perf_counter() - started
    accepted = bool(lean_pass if allow_sorry else lean_verify)

    result = {
        "file": str(path),
        "lean_pass": bool(lean_pass),
        "lean_verify": bool(lean_verify),
        "accepted": accepted,
        "mode": "stage2_allow_sorry" if allow_sorry else "stage3_no_sorry",
        "elapsed_seconds": round(elapsed, 3),
    }
    error_msg = compact_output(output)
    if error_msg:
        result["error_msg"] = error_msg
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return accepted


async def run_once(args: argparse.Namespace, server: LeanServer) -> int:
    assert args.file is not None
    path = resolve_lean_file(args.file)
    accepted = await verify_file(
        server,
        path,
        add_imports=args.add_imports,
        allow_sorry=args.allow_sorry,
    )
    return 0 if accepted else 1


async def run_repl(args: argparse.Namespace, server: LeanServer) -> int:
    current_path = resolve_lean_file(args.file) if args.file else None
    print(
        "Lean file verifier is ready. "
        "Press Enter to re-check the current file, type a .lean path, or type q to quit."
    )
    if current_path is not None:
        await verify_file(
            server,
            current_path,
            add_imports=args.add_imports,
            allow_sorry=args.allow_sorry,
        )

    while True:
        prompt = f"lean-file [{current_path}]: " if current_path else "lean-file: "
        try:
            raw = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        value = raw.strip().strip('"')
        if value.lower() in {"q", "quit", "exit"}:
            return 0
        if value:
            try:
                current_path = resolve_lean_file(value)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
        if current_path is None:
            print("error: provide a .lean file path first", file=sys.stderr)
            continue

        try:
            await verify_file(
                server,
                current_path,
                add_imports=args.add_imports,
                allow_sorry=args.allow_sorry,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)


async def amain(args: argparse.Namespace) -> int:
    mathlib_path = Path(args.mathlib_path).expanduser().resolve()
    if not mathlib_path.is_dir():
        raise RuntimeError(f"--mathlib-path is not a directory: {mathlib_path}")

    temp_root = Path(args.temp_root).expanduser().resolve()
    temp_root.mkdir(parents=True, exist_ok=True)

    server = LeanServer(
        project_path=str(mathlib_path),
        backend=args.backend,
        pool_size=args.pool_size,
        temp_root=str(temp_root),
    )
    try:
        if args.backend == "persistent_lsp":
            print(
                "[lean-file-verifier] initializing persistent Lean LSP "
                f"(mathlib={mathlib_path}, pool_size={args.pool_size}) ..."
            )
            await server.ensure_ready()
            print("[lean-file-verifier] ready.")

        if args.repl:
            return await run_repl(args, server)
        return await run_once(args, server)
    finally:
        await server.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a pure .lean file with the same local LeanServer path used by "
            "step-proof stage2/stage3 validation."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a pure .lean file. In --repl mode, Enter re-checks this file.",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Keep Lean alive and repeatedly verify the current .lean file or newly entered paths.",
    )
    parser.add_argument(
        "--mathlib-path",
        default=str(default_mathlib_path()),
        help="Lean project path used for verification. Defaults to $CZX_ROOT/mathlib4.",
    )
    parser.add_argument(
        "--backend",
        choices=["persistent_lsp", "subprocess"],
        default="persistent_lsp",
        help="Lean backend. persistent_lsp keeps the service warm in --repl mode.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Number of persistent Lean LSP workers.",
    )
    parser.add_argument(
        "--temp-root",
        default=str(default_temp_root()),
        help="Temporary directory for Lean jobs.",
    )
    parser.add_argument(
        "--add-imports",
        action="store_true",
        help="Prepend the default Mathlib/Aesop imports and options if missing.",
    )
    parser.add_argument(
        "--allow-sorry",
        action="store_true",
        help="Accept lean_pass even when sorry remains, matching stage2 formalization checks.",
    )
    args = parser.parse_args()
    if not args.file and not args.repl:
        parser.error("file is required unless --repl is used")
    if args.pool_size < 1:
        parser.error("--pool-size must be >= 1")
    if args.file:
        try:
            args.file = str(resolve_lean_file(args.file))
        except Exception as exc:
            parser.error(str(exc))
    return args


def main() -> int:
    try:
        return asyncio.run(amain(parse_args()))
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# python scripts/verify_lean_file.py /data/run01/scyb202/czx/czx_work/TEMP/lean_jobs/1.lean  --repl