from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


JsonDict = Dict[str, Any]

FDG_GRAPH_MODE = "fdg"
GRAPH_MODES = {FDG_GRAPH_MODE}


def normalize_graph_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return FDG_GRAPH_MODE if normalized in {"", FDG_GRAPH_MODE} else normalized


def record_graph_mode(record: JsonDict) -> str:
    meta = record.get("meta") or {}
    graph = record.get("graph") or {}
    results = record.get("results") or {}

    schema_version = str(meta.get("schema_version", "")).strip().lower()
    if schema_version.startswith("fdg-"):
        return FDG_GRAPH_MODE

    if isinstance(graph.get("facts"), list) or isinstance(results.get("facts"), list):
        return FDG_GRAPH_MODE

    mode = normalize_graph_mode(meta.get("graph_mode"))
    if mode == FDG_GRAPH_MODE:
        return FDG_GRAPH_MODE
    return mode or "unknown"


def ensure_single_graph_mode(records: Iterable[JsonDict], *, source_name: str) -> str:
    seen: set[str] = set()
    for record in records:
        seen.add(record_graph_mode(record))
    if not seen:
        return FDG_GRAPH_MODE
    if len(seen) > 1:
        raise ValueError(f"{source_name} mixes multiple graph_mode values: {sorted(seen)}")
    mode = next(iter(seen))
    if mode != FDG_GRAPH_MODE:
        raise ValueError(f"{source_name} is not an FDG file: graph_mode={mode!r}")
    return mode


def detect_graph_mode_from_jsonl(path: Path) -> str:
    modes: set[str] = set()
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            modes.add(record_graph_mode(json.loads(line)))
    if not modes:
        return FDG_GRAPH_MODE
    if len(modes) > 1:
        raise ValueError(f"{path} mixes multiple graph_mode values: {sorted(modes)}")
    mode = next(iter(modes))
    if mode != FDG_GRAPH_MODE:
        raise ValueError(f"{path} is not an FDG file: graph_mode={mode!r}")
    return mode


def ensure_fdg_jsonl(path: Path) -> None:
    detect_graph_mode_from_jsonl(path)


def graph_items_key(mode: str) -> str:
    if normalize_graph_mode(mode) != FDG_GRAPH_MODE:
        raise ValueError(f"FDG-only runtime does not support graph_mode={mode!r}")
    return "facts"


def extract_record_items(record: JsonDict, source: str) -> Tuple[List[JsonDict], str]:
    mode = record_graph_mode(record)
    key = graph_items_key(mode)
    if source == "results":
        items = list((record.get("results") or {}).get(key) or [])
        if not items:
            items = list((record.get("graph") or {}).get(key) or [])
    elif source == "graph":
        items = list((record.get("graph") or {}).get(key) or [])
        if not items:
            items = list((record.get("results") or {}).get(key) or [])
    else:
        raise ValueError(f"invalid source: {source}")
    return items, mode


def item_id(item: JsonDict) -> str:
    return str(item.get("fact_id") or item.get("id") or "").strip()


def item_dependencies(item: JsonDict) -> List[str]:
    return [str(value).strip() for value in item.get("parent_fact_ids") or [] if str(value).strip()]


def item_text(item: JsonDict) -> str:
    return str(item.get("text") or "").strip()


def item_is_final(mode: str, item: JsonDict) -> bool:
    if normalize_graph_mode(mode) != FDG_GRAPH_MODE:
        raise ValueError(f"FDG-only runtime does not support graph_mode={mode!r}")
    return bool(item.get("is_final_answer", False))
