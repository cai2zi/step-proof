from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


JsonDict = Dict[str, Any]

LEGACY_GRAPH_MODE = "legacy"
FDG_GRAPH_MODE = "fdg"
GRAPH_MODES = {LEGACY_GRAPH_MODE, FDG_GRAPH_MODE}


def normalize_graph_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in GRAPH_MODES:
        return normalized
    return LEGACY_GRAPH_MODE


def record_graph_mode(record: JsonDict) -> str:
    meta = record.get("meta") or {}
    graph = record.get("graph") or {}
    results = record.get("results") or {}

    mode = normalize_graph_mode(meta.get("graph_mode"))
    if mode in GRAPH_MODES and str(meta.get("graph_mode", "")).strip():
        return mode

    schema_version = str(meta.get("schema_version", "")).strip().lower()
    if schema_version.startswith("fdg-"):
        return FDG_GRAPH_MODE

    if isinstance(graph.get("facts"), list) or isinstance(results.get("facts"), list):
        return FDG_GRAPH_MODE

    return LEGACY_GRAPH_MODE


def ensure_single_graph_mode(records: Iterable[JsonDict], *, source_name: str) -> str:
    seen: set[str] = set()
    for record in records:
        seen.add(record_graph_mode(record))
    if not seen:
        return LEGACY_GRAPH_MODE
    if len(seen) > 1:
        raise ValueError(f"{source_name} mixes multiple graph_mode values: {sorted(seen)}")
    return next(iter(seen))


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
        return LEGACY_GRAPH_MODE
    if len(modes) > 1:
        raise ValueError(f"{path} mixes multiple graph_mode values: {sorted(modes)}")
    return next(iter(modes))


def graph_items_key(mode: str) -> str:
    return "facts" if normalize_graph_mode(mode) == FDG_GRAPH_MODE else "nodes"


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
    if isinstance(item.get("parent_fact_ids"), list):
        return [str(value).strip() for value in item.get("parent_fact_ids") or [] if str(value).strip()]
    return [str(value).strip() for value in item.get("dependencies") or [] if str(value).strip()]


def item_text(item: JsonDict) -> str:
    return str(item.get("text") or item.get("statement") or item.get("natural_language") or "").strip()


def item_is_final(mode: str, item: JsonDict) -> bool:
    if normalize_graph_mode(mode) == FDG_GRAPH_MODE:
        return bool(item.get("is_final_answer", False))
    return str(item.get("role", "")).strip() == "final"
