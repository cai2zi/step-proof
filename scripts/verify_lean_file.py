#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MAX_ERROR_ITEMS = 5
MAX_ERROR_TEXT_CHARS = 2000
MAX_MESSAGE_DATA_CHARS = 500


def resolve_lean_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.suffix.lower() != ".lean":
        raise ValueError(f"input must be a pure .lean file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Lean file not found: {path}")
    return path.resolve()


def process_lean_string(lean_string: str) -> str:
    lines = lean_string.split("\n")
    required = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]
    present = {stmt for line in lines for stmt in required if stmt in line.strip()}

    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
            or stripped == ""
            or stripped.startswith("--")
        ):
            insert_at = i + 1
        else:
            break

    for stmt in reversed([stmt for stmt in required if stmt not in present]):
        lines.insert(insert_at, stmt)

    imports_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
            or stripped.startswith("--")
        ):
            imports_end = i + 1
        elif stripped == "":
            continue
        else:
            break

    if imports_end < len(lines) and imports_end > 0 and lines[imports_end].strip() != "":
        lines.insert(imports_end, "")

    return "\n".join(lines).lstrip("\n")


def _truncate_text(text: str, max_chars: int = MAX_ERROR_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n... [truncated {len(text) - max_chars} chars]"


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("severity", "pos", "endPos", "kind"):
        if key in message:
            compact[key] = message[key]
    if "data" in message:
        compact["data"] = _truncate_text(str(message["data"]), MAX_MESSAGE_DATA_CHARS)
    elif "message" in message:
        compact["message"] = _truncate_text(str(message["message"]), MAX_MESSAGE_DATA_CHARS)
    return compact or {
        "message": _truncate_text(json.dumps(message, ensure_ascii=False), MAX_MESSAGE_DATA_CHARS)
    }


def _compact_messages(messages: list[Any], max_items: int = MAX_ERROR_ITEMS) -> dict[str, Any]:
    error_messages = [
        message
        for message in messages
        if isinstance(message, dict) and _message_severity(message) == "error"
    ]
    selected = error_messages if error_messages else [m for m in messages if isinstance(m, dict)]
    selected = selected[:max_items]

    return {
        "shown": [_compact_message(message) for message in selected],
        "shown_count": len(selected),
        "total_count": len(messages),
        "omitted_count": max(0, len(messages) - len(selected)),
    }


def _compact_payload(output: Any) -> Any:
    if isinstance(output, dict):
        compact: dict[str, Any] = {}
        if "messages" in output and isinstance(output["messages"], list):
            compact["messages"] = _compact_messages(output["messages"])
        if "sorries" in output and isinstance(output["sorries"], list):
            sorries = output["sorries"]
            compact["sorries"] = {
                "shown": sorries[:MAX_ERROR_ITEMS],
                "shown_count": min(len(sorries), MAX_ERROR_ITEMS),
                "total_count": len(sorries),
                "omitted_count": max(0, len(sorries) - MAX_ERROR_ITEMS),
            }
        if "response" in output:
            compact["response"] = _truncate_text(
                json.dumps(output["response"], ensure_ascii=False),
                MAX_ERROR_TEXT_CHARS,
            )
        for key in ("error", "message"):
            if key in output:
                compact[key] = _truncate_text(str(output[key]), MAX_ERROR_TEXT_CHARS)
        return compact or _truncate_text(json.dumps(output, ensure_ascii=False), MAX_ERROR_TEXT_CHARS)

    if isinstance(output, list):
        return {
            "shown": output[:MAX_ERROR_ITEMS],
            "shown_count": min(len(output), MAX_ERROR_ITEMS),
            "total_count": len(output),
            "omitted_count": max(0, len(output) - MAX_ERROR_ITEMS),
        }

    return output


def compact_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        text = output.strip()
        return "" if text == "[]" else _truncate_text(text)
    if isinstance(output, (list, tuple, dict)) and not output:
        return ""
    return json.dumps(_compact_payload(output), ensure_ascii=False, indent=2)


def _message_severity(message: dict[str, Any]) -> str:
    severity = message.get("severity", "")
    if isinstance(severity, str):
        return severity.lower()
    if severity == 1:
        return "error"
    if severity == 2:
        return "warning"
    if severity == 3:
        return "info"
    return str(severity).lower()


def _contains_sorry(payload: Any) -> bool:
    try:
        text = json.dumps(payload, ensure_ascii=False).lower()
    except Exception:
        text = str(payload).lower()
    return any(
        marker in text
        for marker in (
            "declaration uses 'sorry'",
            'declaration uses "sorry"',
            "declaration uses `sorry`",
            '"kind":"hassorry"',
            '"kind": "hassorry"',
            "has_sorry",
            "hassorry",
        )
    )


def analyze_kimina_result(result: dict[str, Any]) -> tuple[bool, bool, Any]:
    if result.get("error"):
        return False, False, result.get("error")

    response = result.get("response")
    if response is None:
        return False, False, result

    if isinstance(response, dict) and response.get("message"):
        return False, False, response

    messages = response.get("messages") if isinstance(response, dict) else []
    messages = messages or []
    sorries = response.get("sorries") if isinstance(response, dict) else []
    sorries = sorries or []

    lean_pass = not any(
        isinstance(message, dict) and _message_severity(message) == "error"
        for message in messages
    )
    lean_verify = lean_pass and not sorries and not _contains_sorry(response)

    if lean_verify:
        return lean_pass, lean_verify, None

    details: dict[str, Any] = {}
    if messages:
        details["messages"] = messages
    if sorries:
        details["sorries"] = sorries
    if not details:
        details["response"] = response
    return lean_pass, lean_verify, details


def kimina_check(
    *,
    api_url: str,
    code: str,
    snippet_id: str,
    timeout: int,
    reuse: bool,
    debug: bool,
    api_key: str | None,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/api/check"
    body = {
        "snippets": [{"id": snippet_id, "code": code}],
        "timeout": timeout,
        "debug": debug,
        "reuse": reuse,
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout + 30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"kimina request failed with HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"kimina request failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kimina returned non-JSON response: {raw}") from exc

    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"kimina response has no results: {data}")
    if not isinstance(results[0], dict):
        raise RuntimeError(f"kimina result is not an object: {results[0]}")
    return results[0]


async def verify_file(args: argparse.Namespace, path: Path) -> bool:
    lean_code = path.read_text(encoding="utf-8")
    if args.add_imports:
        lean_code = process_lean_string(lean_code)

    started = time.perf_counter()
    try:
        kimina_result = await asyncio.to_thread(
            kimina_check,
            api_url=args.api_url,
            code=lean_code,
            snippet_id=f"{path.stem}-{time.time_ns()}",
            timeout=args.timeout,
            reuse=not args.no_reuse,
            debug=args.debug,
            api_key=args.api_key,
        )
        lean_pass, lean_verify, output = analyze_kimina_result(kimina_result)
    except Exception as exc:
        lean_pass, lean_verify, output = False, False, str(exc)
    elapsed = time.perf_counter() - started

    accepted = bool(lean_pass if args.allow_sorry else lean_verify)

    result = {
        "file": str(path),
        "lean_pass": bool(lean_pass),
        "lean_verify": bool(lean_verify),
        "accepted": accepted,
        "mode": "stage2_allow_sorry" if args.allow_sorry else "stage3_no_sorry",
        "backend": "kimina",
        "api_url": args.api_url,
        "reuse": not args.no_reuse,
        "elapsed_seconds": round(elapsed, 3),
    }
    error_msg = compact_output(output)
    if error_msg:
        result["error_msg"] = error_msg
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return accepted


async def run_once(args: argparse.Namespace) -> int:
    assert args.file is not None
    accepted = await verify_file(args, resolve_lean_file(args.file))
    return 0 if accepted else 1


async def run_interactive(args: argparse.Namespace) -> int:
    current_path = resolve_lean_file(args.file) if args.file else None
    print(
        "Kimina Lean file verifier is ready. "
        "Press Enter to re-check the current file, type a .lean path, or type q to quit."
    )
    if current_path is not None:
        await verify_file(args, current_path)

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
            await verify_file(args, current_path)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)


async def amain(args: argparse.Namespace) -> int:
    if args.interactive:
        return await run_interactive(args)
    return await run_once(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a pure .lean file through a running kimina-lean-server."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a pure .lean file. In --interactive mode, Enter re-checks this file.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("KIMINA_API_URL", "http://localhost:8000"),
        help="kimina-lean-server base URL. Defaults to $KIMINA_API_URL or http://localhost:8000.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KIMINA_API_KEY") or os.environ.get("LEAN_SERVER_API_KEY"),
        help="Optional bearer token. Defaults to $KIMINA_API_KEY or $LEAN_SERVER_API_KEY.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Lean command timeout sent to kimina, in seconds.",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Disable kimina server-side REPL/header reuse for this request.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ask kimina to include diagnostics in the raw response.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Repeatedly verify the current .lean file or newly entered paths.",
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
    if not args.file and not args.interactive:
        parser.error("file is required unless --interactive is used")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
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


# Example:
# python scripts/verify_lean_file.py /data/run01/scyb202/czx/TEMP/lean_jobs/1.lean --api-url http://localhost:8000
