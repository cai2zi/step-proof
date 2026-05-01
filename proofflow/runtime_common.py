from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

JsonDict = Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def append_jsonl(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("meta", {}).get("record_id")
            if rid:
                done.add(str(rid))
    return done


def extract_last_lean_block(text_input: str) -> str:
    from .lean_check import process_lean_string

    matches = re.findall(r"```lean4\s*\n(.*?)\n```", text_input, re.DOTALL)
    if matches:
        return process_lean_string(matches[-1].strip())

    fallback = _extract_unfenced_lean(text_input)
    if fallback:
        return process_lean_string(fallback)

    raise ValueError("No Lean 4 code block or Lean declaration found.")


def _extract_unfenced_lean(text_input: str) -> str:
    lines = text_input.strip().splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
            or stripped.startswith("lemma ")
            or stripped.startswith("theorem ")
        ):
            start = idx
            break
    if start is None:
        return ""

    selected: List[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        selected.append(line)
    return "\n".join(selected).strip()
