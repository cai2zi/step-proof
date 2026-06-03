from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping


JsonDict = Dict[str, Any]

PARENT_ONLY = "parent_only"
PROBLEM_PARENT = "problem_parent"
PROBLEM_PREFIX = "problem_prefix"
PROBLEM_FULL_GRAPH = "problem_full_graph"
PROBLEM_COT_FULL_GRAPH = "problem_cot_full_graph"

FORMALIZER_CONTEXT_MODES = {
    PARENT_ONLY,
    PROBLEM_PARENT,
    PROBLEM_PREFIX,
    PROBLEM_FULL_GRAPH,
    PROBLEM_COT_FULL_GRAPH,
}


def normalize_context_mode(mode: str | None) -> str:
    value = str(mode or PARENT_ONLY).strip()
    aliases = {
        "c0": PARENT_ONLY,
        "c0_parent": PARENT_ONLY,
        "c1": PROBLEM_PARENT,
        "c1_problem_parent": PROBLEM_PARENT,
        "c2": PROBLEM_PREFIX,
        "c2_problem_prefix": PROBLEM_PREFIX,
        "c3": PROBLEM_FULL_GRAPH,
        "c3_problem_full_graph": PROBLEM_FULL_GRAPH,
        "c4": PROBLEM_COT_FULL_GRAPH,
        "c4_problem_cot_full_graph": PROBLEM_COT_FULL_GRAPH,
    }
    value = aliases.get(value, value)
    if value not in FORMALIZER_CONTEXT_MODES:
        allowed = ", ".join(sorted(FORMALIZER_CONTEXT_MODES))
        raise ValueError(f"Unknown formalizer context mode {mode!r}; allowed: {allowed}")
    return value


def _facts_from_record(record: Mapping[str, Any]) -> List[JsonDict]:
    runtime_facts = record.get("facts")
    if isinstance(runtime_facts, Mapping):
        return [dict(value) for value in runtime_facts.values()]
    graph_facts = (record.get("graph") or {}).get("facts") if isinstance(record.get("graph"), Mapping) else None
    return [dict(value) for value in graph_facts or []]


def _fact_lookup(record: Mapping[str, Any]) -> Dict[str, JsonDict]:
    return {str(fact.get("fact_id")): fact for fact in _facts_from_record(record)}


def _topo_ids(record: Mapping[str, Any], facts: Mapping[str, JsonDict]) -> List[str]:
    graph = record.get("graph") if isinstance(record.get("graph"), Mapping) else {}
    order = [str(item) for item in (graph.get("topo_order") or []) if str(item) in facts]
    if order:
        return order
    return list(facts.keys())


def _compact_fact(fact: Mapping[str, Any]) -> JsonDict:
    return {
        "fact_id": fact.get("fact_id", ""),
        "text": fact.get("text", ""),
        "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
        "origin": fact.get("origin", ""),
        "is_final_answer": bool(fact.get("is_final_answer", False)),
        "skip": int(fact.get("skip", 1) or 0),
    }


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _section(title: str, content: Any) -> str:
    if isinstance(content, str):
        body = content.strip()
    else:
        body = _json_block(content)
    return f"{title}:\n{body}" if body else f"{title}:\n<empty>"


def _parent_facts(facts: Mapping[str, JsonDict], current: Mapping[str, Any]) -> List[JsonDict]:
    parents: List[JsonDict] = []
    for parent_id in current.get("parent_fact_ids") or []:
        parent = facts.get(str(parent_id))
        if parent is not None:
            parents.append(_compact_fact(parent))
    return parents


def _graph_prefix(
    record: Mapping[str, Any],
    facts: Mapping[str, JsonDict],
    current_fact_id: str,
) -> List[JsonDict]:
    prefix: List[JsonDict] = []
    for fact_id in _topo_ids(record, facts):
        if fact_id == current_fact_id:
            break
        prefix.append(_compact_fact(facts[fact_id]))
    return prefix


def _full_graph(record: Mapping[str, Any], facts: Mapping[str, JsonDict]) -> List[JsonDict]:
    return [_compact_fact(facts[fact_id]) for fact_id in _topo_ids(record, facts)]


def build_formalizer_context(
    record: Mapping[str, Any],
    fact_id: str,
    *,
    mode: str = PARENT_ONLY,
) -> str:
    """Build the natural-language context supplied to the formalizer.

    The function is intentionally pure and JSON-oriented so ablations can be
    diffed, tested, and logged without touching the stage runner.
    """

    resolved_mode = normalize_context_mode(mode)
    facts = _fact_lookup(record)
    current = facts.get(str(fact_id))
    if current is None:
        raise KeyError(f"Unknown fact_id: {fact_id}")

    input_payload = record.get("input") if isinstance(record.get("input"), Mapping) else {}
    problem = str(input_payload.get("problem", "")).strip()
    raw_cot = str(input_payload.get("raw_cot", "")).strip()
    current_fact = _compact_fact(current)
    parents = _parent_facts(facts, current)

    sections: List[str] = []
    if resolved_mode in {
        PROBLEM_PARENT,
        PROBLEM_PREFIX,
        PROBLEM_FULL_GRAPH,
        PROBLEM_COT_FULL_GRAPH,
    }:
        sections.append(_section("Problem", problem))
    if resolved_mode == PROBLEM_COT_FULL_GRAPH:
        sections.append(_section("Full CoT", raw_cot))

    if resolved_mode == PROBLEM_PREFIX:
        sections.append(
            _section(
                "Graph prefix before current node",
                _graph_prefix(record, facts, str(fact_id)),
            )
        )
    elif resolved_mode in {PROBLEM_FULL_GRAPH, PROBLEM_COT_FULL_GRAPH}:
        sections.append(_section("Full graph", _full_graph(record, facts)))

    sections.append(_section("Current node", current_fact))
    if resolved_mode in {PARENT_ONLY, PROBLEM_PARENT}:
        sections.append(_section("Parent node(s)", parents))
    if resolved_mode in {PROBLEM_FULL_GRAPH, PROBLEM_COT_FULL_GRAPH}:
        sections.append(_section("Current node id", str(fact_id)))

    return "\n\n".join(sections)


def iter_supported_context_modes() -> Iterable[str]:
    return sorted(FORMALIZER_CONTEXT_MODES)
