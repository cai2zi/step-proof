from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, TypeAlias

from .lean_check import process_lean_string
from .node_schema import (
    ROLE_CLAIM,
    ROLE_CONDITION,
    ROLE_CONTEXT,
    ROLE_FINAL,
    infer_role,
)
from .prompt_builder import build_chat_messages
from .utils import remove_imports

FORM_TERMINAL = {"success", "failed", "skipped"}
PROVE_TERMINAL = {"success", "failed", "skipped"}

JsonDict: TypeAlias = Dict[str, Any]
NodeState: TypeAlias = Dict[str, Any]
RecordState: TypeAlias = Dict[str, Any]
StageTask: TypeAlias = Dict[str, Any]
StageResult: TypeAlias = Dict[str, Any]


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
            if not line:
                continue
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
    matches = re.findall(r"```lean4\s*\n(.*?)\n```", text_input, re.DOTALL)
    if not matches:
        raise ValueError("No Lean 4 code block found.")
    return process_lean_string(matches[-1].strip())


def role_of(node: NodeState, id_schema_mode: str) -> str:
    return node.get("role") or infer_role(
        node.get("id", ""),
        node.get("node_type"),
        mode=id_schema_mode,
    )


def needs_verification(node: NodeState) -> int:
    return int(node.get("needs_verification", 0) or 0)


def form_is_skipped(node: NodeState, id_schema_mode: str) -> bool:
    role = role_of(node, id_schema_mode)
    return role == ROLE_CONDITION or (
        role == ROLE_CONTEXT and needs_verification(node) != 1
    )


def blocks_children(node: NodeState, id_schema_mode: str) -> bool:
    return not form_is_skipped(node, id_schema_mode)


def should_prove(node: NodeState, id_schema_mode: str) -> bool:
    role = role_of(node, id_schema_mode)
    return role in {ROLE_CLAIM, ROLE_FINAL} or (
        role == ROLE_CONTEXT and needs_verification(node) == 1
    )


def empty_formalization(skipped: bool = False) -> JsonDict:
    payload = {
        "lean_code": "",
        "lean_pass": False,
        "error_msg": [],
        "tries": 0,
        "attempt_history": [],
    }
    if skipped:
        payload["lean_pass"] = True
        payload["skipped"] = True
    return payload


def empty_solver(skipped: bool = False) -> JsonDict:
    payload = {
        "lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": [],
        "tries": 0,
        "attempt_history": [],
    }
    if skipped:
        payload["skipped"] = True
    return payload


def build_dependency_context(
    node: NodeState,
    nodes: Dict[str, NodeState],
) -> str:
    dependencies = list(node.get("dependencies") or [])
    if not dependencies:
        return ""

    intro = (
        f"\n\n This proof step depend on previous proof steps, namely steps {dependencies}.\n"
        "Please make use use of their formal lean4 code, which contains relevant lean4 "
        "hypothesis and type declarations you may use:"
    )
    parts: List[str] = []
    for dep_id in dependencies:
        dep = nodes.get(dep_id)
        if dep is None:
            continue
        formalization = dep.get("formalization") or {}
        if formalization.get("lean_code") and formalization.get("lean_pass"):
            parts.append("\n")
            parts.append(remove_imports(formalization["lean_code"]))
        else:
            parts.append(
                "\nDependency step "
                f"{dep_id} is provided in natural language: \"{dep.get('statement', '')}\". "
                "Please formalize it as part of your current lemma's hypotheses."
            )

    if not parts:
        return ""

    footer = (
        "\nFocus on the original formalization task I gave you and use the previous Lean codes, "
        "extra context, type declarations, variables domains, etc. You can assume the "
        "information is correct. Make use of it!"
    )
    return (intro + "\n".join(parts) + footer).strip()


def build_form_messages(
    record: RecordState,
    node: NodeState,
) -> List[Dict[str, str]]:
    role = role_of(node, record["meta"]["id_schema_mode"])
    lemma_header = f"theorem {node['id']}" if role == ROLE_FINAL else f"lemma {node['id']}"
    return build_chat_messages(
        "calc",
        "formalize_claim",
        lemma_header=lemma_header,
        statement=node["statement"],
        dependencies=list(node.get("dependencies") or []),
        dependency_context_block=build_dependency_context(node, record["nodes"]),
    )


def build_prove_messages(node: NodeState) -> List[Dict[str, str]]:
    messages = build_chat_messages(
        "calc",
        "prove",
        statement=node["statement"],
        lean_code=(node.get("formalization") or {}).get("lean_code", ""),
    )
    formalization = node.get("formalization") or {}
    if formalization.get("lean_code") and not formalization.get("lean_pass"):
        messages[1]["content"] += (
            "\n\nThe previous Lean4 code I sent you contains errors. Please take that into account."
        )
    return messages


def stage2_record_terminal(record: RecordState) -> bool:
    return all(node["form_status"] in FORM_TERMINAL for node in record["nodes"].values())


def stage3_record_terminal(record: RecordState) -> bool:
    return all(
        node.get("prove_status", "pending") in PROVE_TERMINAL
        for node in record["nodes"].values()
    )


def _node_result_base(node: NodeState) -> JsonDict:
    return {
        "id": node["id"],
        "role": node["role"],
        "node_type": node.get("node_type", ""),
        "natural_language": node.get("natural_language", ""),
        "statement": node.get("statement", ""),
        "dependencies": list(node.get("dependencies") or []),
        "needs_verification": node.get("needs_verification"),
    }


def _form_counts(record: RecordState) -> Dict[str, int]:
    counts = defaultdict(int)
    for node in record["nodes"].values():
        counts[node["form_status"]] += 1
    return dict(counts)


def _prove_counts(record: RecordState) -> Dict[str, int]:
    counts = defaultdict(int)
    for node in record["nodes"].values():
        counts[node.get("prove_status", "pending")] += 1
    return dict(counts)


def stage2_record_summary(record: RecordState) -> JsonDict:
    any_failed = any(
        node["form_status"] == "failed" for node in record["nodes"].values()
    )
    status = "completed_with_failures" if any_failed else "completed"
    return {
        "record_status": status,
        "updated_at": utc_now_iso(),
        "form_stats": _form_counts(record),
    }


def stage3_record_summary(record: RecordState) -> JsonDict:
    any_failed = any(
        node["form_status"] == "failed" or node.get("prove_status") == "failed"
        for node in record["nodes"].values()
    )
    status = "completed_with_failures" if any_failed else "completed"
    return {
        "record_status": status,
        "updated_at": utc_now_iso(),
        "form_stats": _form_counts(record),
        "prove_stats": _prove_counts(record),
    }


def stage2_result_nodes(record: RecordState) -> List[JsonDict]:
    order = record["graph"].get("topo_order") or list(record["nodes"].keys())
    nodes: List[JsonDict] = []
    for node_id in order:
        node = record["nodes"].get(node_id)
        if node is None:
            continue
        payload = _node_result_base(node)
        payload["form_status"] = node["form_status"]
        payload["formalization"] = node.get("formalization") or empty_formalization()
        nodes.append(payload)
    return nodes


def stage3_result_nodes(record: RecordState) -> List[JsonDict]:
    order = record["graph"].get("topo_order") or list(record["nodes"].keys())
    nodes: List[JsonDict] = []
    for node_id in order:
        node = record["nodes"].get(node_id)
        if node is None:
            continue
        payload = _node_result_base(node)
        payload["form_status"] = node["form_status"]
        payload["formalization"] = node.get("formalization") or empty_formalization()
        payload["prove_status"] = node.get("prove_status", "skipped")
        payload["solved_lemma"] = node.get("solved_lemma") or empty_solver(skipped=True)
        nodes.append(payload)
    return nodes


def stage2_checkpoint_payload(record: RecordState) -> JsonDict:
    return {
        "meta": record["meta"],
        "input": record["input"],
        "graph": record["graph"],
        "execution": {
            "record_status": record["record_status"],
            "created_at": record["created_at"],
            "updated_at": utc_now_iso(),
        },
        "runtime_nodes": {
            node_id: {
                "id": node["id"],
                "role": node["role"],
                "node_type": node.get("node_type", ""),
                "natural_language": node.get("natural_language", ""),
                "statement": node.get("statement", ""),
                "dependencies": list(node.get("dependencies") or []),
                "needs_verification": node.get("needs_verification"),
                "blocking_parents": list(node.get("blocking_parents") or []),
                "blocking_remaining": int(node.get("blocking_remaining", 0)),
                "successors": list(node.get("successors") or []),
                "form_status": node["form_status"],
                "form_retries_used": int(node.get("form_retries_used", 0)),
                "formalization": node.get("formalization") or empty_formalization(),
                "form_messages": node.get("form_messages") or [],
                "form_attempt_history": node.get("form_attempt_history") or [],
            }
            for node_id, node in record["nodes"].items()
        },
    }


def stage3_checkpoint_payload(record: RecordState) -> JsonDict:
    return {
        "meta": record["meta"],
        "input": record["input"],
        "graph": record["graph"],
        "execution": {
            "record_status": record["record_status"],
            "created_at": record["created_at"],
            "updated_at": utc_now_iso(),
        },
        "runtime_nodes": {
            node_id: {
                "id": node["id"],
                "role": node["role"],
                "node_type": node.get("node_type", ""),
                "natural_language": node.get("natural_language", ""),
                "statement": node.get("statement", ""),
                "dependencies": list(node.get("dependencies") or []),
                "needs_verification": node.get("needs_verification"),
                "form_status": node["form_status"],
                "formalization": node.get("formalization") or empty_formalization(),
                "prove_status": node.get("prove_status", "skipped"),
                "prove_retries_used": int(node.get("prove_retries_used", 0)),
                "solved_lemma": node.get("solved_lemma") or empty_solver(skipped=True),
                "prove_messages": node.get("prove_messages") or [],
                "prove_attempt_history": node.get("prove_attempt_history") or [],
            }
            for node_id, node in record["nodes"].items()
        },
    }


def stage2_final_payload(record: RecordState) -> JsonDict:
    meta = dict(record["meta"])
    meta["schema_version"] = "graph-form-v1"
    meta["stage2_created_at"] = record["created_at"]
    meta["stage2_updated_at"] = utc_now_iso()
    return {
        "meta": meta,
        "input": record["input"],
        "graph": record["graph"],
        "execution": stage2_record_summary(record),
        "results": {
            "nodes": stage2_result_nodes(record),
        },
    }


def stage3_final_payload(record: RecordState) -> JsonDict:
    meta = dict(record["meta"])
    meta["schema_version"] = "graph-formprove-v1"
    meta["stage3_created_at"] = record["created_at"]
    meta["stage3_updated_at"] = utc_now_iso()
    return {
        "meta": meta,
        "input": record["input"],
        "graph": record["graph"],
        "execution": stage3_record_summary(record),
        "results": {
            "nodes": stage3_result_nodes(record),
        },
    }


def fresh_record_state(raw: JsonDict) -> RecordState:
    meta = dict(raw.get("meta") or {})
    meta.setdefault("id_schema_mode", "calc")
    graph = dict(raw.get("graph") or {})
    graph_nodes = list(graph.get("nodes") or [])
    node_lookup = {node["id"]: dict(node) for node in graph_nodes}
    successors: Dict[str, List[str]] = defaultdict(list)
    for node in graph_nodes:
        for dep_id in node.get("dependencies") or []:
            if dep_id in node_lookup:
                successors[dep_id].append(node["id"])

    nodes: Dict[str, NodeState] = {}
    for node in graph_nodes:
        role = role_of(node, meta["id_schema_mode"])
        state = dict(node)
        state["role"] = role
        state["successors"] = list(successors.get(node["id"], []))
        state["blocking_parents"] = []
        state["blocking_remaining"] = 0
        state["form_retries_used"] = 0
        state["form_attempt_history"] = []
        state["formalization"] = empty_formalization(
            form_is_skipped(node, meta["id_schema_mode"])
        )
        state["form_messages"] = []
        state["_form_enqueued"] = False
        state["form_status"] = (
            "skipped" if form_is_skipped(node, meta["id_schema_mode"]) else "pending"
        )
        nodes[node["id"]] = state

    for node in nodes.values():
        blocking_parents = [
            dep_id
            for dep_id in node.get("dependencies") or []
            if dep_id in nodes and blocks_children(nodes[dep_id], meta["id_schema_mode"])
        ]
        node["blocking_parents"] = blocking_parents
        node["blocking_remaining"] = sum(
            1
            for dep_id in blocking_parents
            if nodes[dep_id]["form_status"] not in FORM_TERMINAL
        )

    return {
        "meta": meta,
        "input": dict(raw.get("input") or {}),
        "graph": graph,
        "nodes": nodes,
        "created_at": utc_now_iso(),
        "record_status": "running",
    }


def restore_record_state(raw: JsonDict) -> RecordState:
    record = {
        "meta": dict(raw.get("meta") or {}),
        "input": dict(raw.get("input") or {}),
        "graph": dict(raw.get("graph") or {}),
        "nodes": {},
        "created_at": raw.get("execution", {}).get("created_at") or utc_now_iso(),
        "record_status": raw.get("execution", {}).get("record_status") or "running",
    }
    record["meta"].setdefault("id_schema_mode", "calc")
    for node_id, node in (raw.get("runtime_nodes") or {}).items():
        restored = dict(node)
        if restored.get("form_status") == "running":
            restored["form_status"] = "pending"
        restored["_form_enqueued"] = False
        record["nodes"][node_id] = restored
    return record


def fresh_stage3_record_state(raw: JsonDict) -> RecordState:
    meta = dict(raw.get("meta") or {})
    meta.setdefault("id_schema_mode", "calc")
    result_nodes = list((raw.get("results") or {}).get("nodes") or [])
    graph = dict(raw.get("graph") or {})
    graph_nodes = list(graph.get("nodes") or [])
    graph_lookup = {node.get("id"): dict(node) for node in graph_nodes if node.get("id")}

    nodes: Dict[str, NodeState] = {}
    for node in result_nodes:
        node_id = node["id"]
        merged = dict(graph_lookup.get(node_id, {}))
        merged.update(dict(node))
        merged["role"] = role_of(merged, meta["id_schema_mode"])
        merged["form_status"] = merged.get("form_status", "pending")
        merged["formalization"] = merged.get("formalization") or empty_formalization()
        merged["prove_retries_used"] = 0
        merged["prove_attempt_history"] = []
        merged["prove_messages"] = []
        merged["_prove_enqueued"] = False

        if not should_prove(merged, meta["id_schema_mode"]):
            merged["prove_status"] = "skipped"
            merged["solved_lemma"] = empty_solver(skipped=True)
        elif (merged.get("formalization") or {}).get("lean_code"):
            merged["prove_status"] = "pending"
            merged["solved_lemma"] = empty_solver()
        else:
            merged["prove_status"] = "skipped"
            merged["solved_lemma"] = empty_solver(skipped=True)
        nodes[node_id] = merged

    return {
        "meta": meta,
        "input": dict(raw.get("input") or {}),
        "graph": graph,
        "nodes": nodes,
        "created_at": utc_now_iso(),
        "record_status": "running",
    }


def restore_stage3_record_state(raw: JsonDict) -> RecordState:
    record = {
        "meta": dict(raw.get("meta") or {}),
        "input": dict(raw.get("input") or {}),
        "graph": dict(raw.get("graph") or {}),
        "nodes": {},
        "created_at": raw.get("execution", {}).get("created_at") or utc_now_iso(),
        "record_status": raw.get("execution", {}).get("record_status") or "running",
    }
    record["meta"].setdefault("id_schema_mode", "calc")
    for node_id, node in (raw.get("runtime_nodes") or {}).items():
        restored = dict(node)
        if restored.get("prove_status") == "running":
            restored["prove_status"] = "pending"
        restored["_prove_enqueued"] = False
        record["nodes"][node_id] = restored
    return record
