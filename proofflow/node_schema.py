from typing import Any, Dict, List


ROLE_CONDITION = "condition"
ROLE_CONTEXT = "context"
ROLE_CLAIM = "claim"
ROLE_FINAL = "final"
ROLE_UNKNOWN = "unknown"


def infer_role(node_id: str, node_type: str | None = None, mode: str = "auto") -> str:
    """Infer semantic role from id prefix and optional node_type."""
    node_id = (node_id or "").strip()
    node_type = (node_type or "").strip().lower()

    use_calc = mode in {"calc", "auto"}
    use_legacy = mode in {"legacy", "auto"}
    lid = node_id.lower()

    # Prefix wins over node_type so LLM mislabels (e.g. fa_1 + claim) do not override ids.
    if use_calc:
        if lid.startswith("fa_") or lid.startswith("ts_"):
            return ROLE_FINAL
        if lid.startswith("pc_"):
            return ROLE_CONDITION
        if lid.startswith("ctx_"):
            return ROLE_CONTEXT
        if lid.startswith("c_"):
            return ROLE_CLAIM

    if use_legacy:
        if lid.startswith("tc_"):
            return ROLE_CONDITION
        if lid.startswith("def_"):
            return ROLE_CONTEXT
        if lid.startswith("l"):
            return ROLE_CLAIM
        if lid.startswith("ts_"):
            return ROLE_FINAL

    if node_type in {"problem_condition"}:
        return ROLE_CONDITION
    if node_type in {"context"}:
        return ROLE_CONTEXT
    if node_type in {"claim"}:
        return ROLE_CLAIM
    if node_type in {"final_answer"}:
        return ROLE_FINAL

    return ROLE_UNKNOWN


def is_structural_final(node_id: str, mode: str = "auto") -> bool:
    """
    True if the id prefix marks a final/theorem node, ignoring node_type.
    Used so mis-tagged nodes (e.g. fa_* with node_type=claim) are not treated as orphans.
    Prefix match is case-insensitive (FA_1 vs fa_1).
    """
    lid = (node_id or "").strip().lower()
    use_calc = mode in {"calc", "auto"}
    use_legacy = mode in {"legacy", "auto"}
    if use_calc and (lid.startswith("fa_") or lid.startswith("ts_")):
        return True
    if use_legacy and lid.startswith("ts_"):
        return True
    return False


def is_final(node_id: str, node_type: str | None = None, mode: str = "auto") -> bool:
    return infer_role(node_id, node_type=node_type, mode=mode) == ROLE_FINAL


def is_context(node_id: str, node_type: str | None = None, mode: str = "auto") -> bool:
    return infer_role(node_id, node_type=node_type, mode=mode) == ROLE_CONTEXT


def is_condition(node_id: str, node_type: str | None = None, mode: str = "auto") -> bool:
    return infer_role(node_id, node_type=node_type, mode=mode) == ROLE_CONDITION


def is_claim(node_id: str, node_type: str | None = None, mode: str = "auto") -> bool:
    return infer_role(node_id, node_type=node_type, mode=mode) == ROLE_CLAIM


def normalize_node(raw_node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize prompt outputs to internal schema keys."""
    normalized = dict(raw_node)
    if "id" in normalized and isinstance(normalized["id"], str):
        normalized["id"] = normalized["id"].strip()
    if "natural_language" not in normalized:
        normalized["natural_language"] = normalized.get("source_text", "")
    normalized.setdefault("dependencies", [])
    if normalized.get("node_type") is None:
        normalized["node_type"] = ""
    return normalized


def normalize_nodes(raw_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_payload, dict):
        return [normalize_node(raw_payload)]
    if not isinstance(raw_payload, list):
        raise ValueError("DAG payload must be a list of node objects or a single object.")
    return [normalize_node(item) for item in raw_payload]
