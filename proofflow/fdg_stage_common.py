from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

from .fdg_graph import FDGDocument, build_proof_obligation_from_fact, strip_think_blocks
from .prompt_builder import build_chat_messages
from .runtime_common import utc_now_iso


JsonDict = Dict[str, Any]
FactState = Dict[str, Any]
RecordState = Dict[str, Any]

FORM_TERMINAL = {"success", "failed", "skipped"}
PROVE_TERMINAL = {"success", "failed", "skipped"}
FORMALIZER_CONTEXT_MODES = {
    "c0_parent",
    "c1_problem_parent",
    "c2_problem_prefix",
    "c3_problem_full_graph",
    "c4_problem_cot_full_graph",
}


def fdg_empty_formalization(skipped: bool = False) -> JsonDict:
    payload = {
        "lean_code": "",
        "lean_pass": False,
        "error_msg": [],
        "tries": 0,
    }
    if skipped:
        payload["lean_pass"] = True
        payload["skipped"] = True
    return payload


def fdg_empty_solver(skipped: bool = False) -> JsonDict:
    payload = {
        "lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": [],
        "tries": 0,
        "conversation_raw": [],
    }
    if skipped:
        payload["skipped"] = True
    return payload


def _fdg_document_from_stage1(raw: JsonDict) -> FDGDocument:
    return FDGDocument.model_validate(
        {
            "problem_id": str((raw.get("meta") or {}).get("record_id", "")).strip(),
            "problem_text": str((raw.get("input") or {}).get("problem", "")).strip(),
            "facts": list((raw.get("graph") or {}).get("facts") or []),
        }
    )


def _ordered_fact_ids(record: RecordState) -> List[str]:
    order = list((record.get("graph") or {}).get("topo_order") or [])
    if order:
        return [fact_id for fact_id in order if fact_id in record["facts"]]
    return list(record["facts"].keys())


def _coerce_skip(value: Any, default: int = 1) -> int:
    if value is None:
        return 0 if default == 0 else 1
    try:
        return 0 if int(value) == 0 else 1
    except (TypeError, ValueError):
        return 0 if default == 0 else 1


def _ensure_skip_flags(facts: Dict[str, FactState]) -> None:
    for fact in facts.values():
        fact["skip"] = _coerce_skip(fact.get("skip"), default=1)


def fdg_fact_should_execute(fact: FactState) -> bool:
    return _coerce_skip(fact.get("skip"), default=1) == 0


def _split_lean_header_body(lean_code: str) -> Dict[str, str]:
    header_lines: List[str] = []
    body_lines: List[str] = []
    in_body = False
    for line in lean_code.splitlines():
        stripped = line.strip()
        if not in_body and (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
            or stripped == ""
        ):
            header_lines.append(line)
            continue
        in_body = True
        body_lines.append(line)
    return {
        "lean_header": "\n".join(header_lines).strip(),
        "lean_body": "\n".join(body_lines).strip(),
    }


def _formalizer_record_id(record: RecordState) -> str:
    return str((record.get("meta") or {}).get("record_id") or "<unknown>").strip()


def _formalizer_fact_id(fact: FactState) -> str:
    return str(fact.get("fact_id") or "<unknown>").strip()


def _semantic_fact_payload(fact: FactState) -> JsonDict:
    return {
        "fact_id": str(fact.get("fact_id") or ""),
        "text": str(fact.get("text") or ""),
        "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
        "origin": str(fact.get("origin") or ""),
        "is_final_answer": bool(fact.get("is_final_answer", False)),
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _require_formalizer_record(record: RecordState | None, fact: FactState) -> RecordState:
    if record is None:
        raise ValueError(
            f"Formalizer context for fact {_formalizer_fact_id(fact)} requires the full record."
        )
    return record


def _formalizer_problem(record: RecordState, fact: FactState) -> str:
    problem = str((record.get("input") or {}).get("problem") or "").strip()
    if not problem:
        raise ValueError(
            f"Record {_formalizer_record_id(record)} fact {_formalizer_fact_id(fact)} "
            "requires a non-empty input.problem for the selected formalizer context mode."
        )
    return problem


def _formalizer_ordered_facts(record: RecordState, fact: FactState) -> List[FactState]:
    facts = record.get("facts") or {}
    current_fact_id = _formalizer_fact_id(fact)
    if current_fact_id not in facts:
        raise ValueError(
            f"Record {_formalizer_record_id(record)} does not contain current fact "
            f"{current_fact_id!r} in its runtime facts."
        )
    ordered_ids = _ordered_fact_ids(record)
    if current_fact_id not in ordered_ids:
        raise ValueError(
            f"Record {_formalizer_record_id(record)} fact {current_fact_id} is missing "
            "from graph.topo_order."
        )
    missing_fact_ids = [fact_id for fact_id in facts if fact_id not in ordered_ids]
    if missing_fact_ids:
        raise ValueError(
            f"Record {_formalizer_record_id(record)} graph.topo_order is missing runtime facts: "
            f"{missing_fact_ids}."
        )
    return [facts[fact_id] for fact_id in ordered_ids]


def build_fdg_form_statement(
    fact: FactState,
    *,
    record: RecordState | None = None,
    context_mode: str = "c0_parent",
) -> str:
    if context_mode not in FORMALIZER_CONTEXT_MODES:
        raise ValueError(
            f"Unknown formalizer_context_mode={context_mode!r}; "
            f"expected one of {sorted(FORMALIZER_CONTEXT_MODES)}."
        )

    proof_obligation = fact.get("proof_obligation") or {}
    obligation = str(proof_obligation.get("informal_statement_content", "")).strip()
    if context_mode == "c0_parent":
        return obligation

    full_record = _require_formalizer_record(record, fact)
    problem = _formalizer_problem(full_record, fact)
    sections = [f"Original problem:\n{problem}"]
    if context_mode == "c1_problem_parent":
        sections.append(f"Target proof obligation:\n{obligation}")
        return "\n\n".join(sections)

    ordered_facts = _formalizer_ordered_facts(full_record, fact)
    current_fact_id = _formalizer_fact_id(fact)
    current_index = next(
        index for index, ordered_fact in enumerate(ordered_facts)
        if _formalizer_fact_id(ordered_fact) == current_fact_id
    )

    if context_mode == "c2_problem_prefix":
        prefix = [_semantic_fact_payload(item) for item in ordered_facts[:current_index]]
        sections.extend(
            [
                f"Graph prefix before current node:\n{_stable_json(prefix)}",
                f"Current node:\n{_stable_json(_semantic_fact_payload(fact))}",
            ]
        )
    else:
        if context_mode == "c4_problem_cot_full_graph":
            raw_cot = str((full_record.get("input") or {}).get("raw_cot") or "")
            visible_cot = strip_think_blocks(raw_cot)
            sections.append(
                "Visible solution or chain of thought:"
                + (f"\n{visible_cot}" if visible_cot else "")
            )
        full_graph = [_semantic_fact_payload(item) for item in ordered_facts]
        sections.extend(
            [
                f"Full graph:\n{_stable_json(full_graph)}",
                f"Current node id:\n{current_fact_id}",
            ]
        )

    sections.append(f"Target proof obligation:\n{obligation}")
    return "\n\n".join(sections)


def build_fdg_form_messages(
    fact: FactState,
    *,
    record: RecordState | None = None,
    context_mode: str = "c0_parent",
    prompt_name: str = "formalize_obligation",
) -> List[Dict[str, str]]:
    proof_obligation = fact.get("proof_obligation") or {}
    problem_name = str(proof_obligation.get("problem_name") or f"prove_{fact['fact_id']}").strip()
    lemma_keyword = "theorem" if fact.get("is_final_answer") else "lemma"
    return build_chat_messages(
        "formalize_obligation",
        prompt_name=prompt_name,
        lemma_header=f"{lemma_keyword} {problem_name}",
        paper_theorem_name="test",
        informal_statement_content=build_fdg_form_statement(
            fact,
            record=record,
            context_mode=context_mode,
        ),
    )


def build_fdg_prove_messages(
    fact: FactState,
    *,
    prompt_name: str = "prove",
) -> List[Dict[str, str]]:
    proof_obligation = fact.get("proof_obligation") or {}
    lean_code = (fact.get("formalization") or {}).get("lean_code", "")
    messages = build_chat_messages(
        "prove",
        prompt_name=prompt_name,
        statement=str(proof_obligation.get("informal_statement_content", "")).strip(),
        lean_code=lean_code,
        **_split_lean_header_body(lean_code),
    )
    formalization = fact.get("formalization") or {}
    if formalization.get("lean_code") and not formalization.get("lean_pass"):
        messages[-1]["content"] += (
            "\n\nThe previous Lean4 code I sent you contains errors. Please take that into account."
        )
    return messages


def fdg_stage2_record_terminal(record: RecordState) -> bool:
    return all(fact["form_status"] in FORM_TERMINAL for fact in record["facts"].values())


def fdg_stage3_record_terminal(record: RecordState) -> bool:
    return all(fact.get("prove_status", "pending") in PROVE_TERMINAL for fact in record["facts"].values())


def fdg_record_summary(record: RecordState, *, include_prove: bool) -> JsonDict:
    form_counts = defaultdict(int)
    prove_counts = defaultdict(int)
    any_failed = False
    for fact in record["facts"].values():
        form_counts[fact["form_status"]] += 1
        any_failed = any_failed or fact["form_status"] == "failed"
        if include_prove:
            status = fact.get("prove_status", "pending")
            prove_counts[status] += 1
            any_failed = any_failed or status == "failed"
    payload: JsonDict = {
        "record_status": "completed_with_failures" if any_failed else "completed",
        "updated_at": utc_now_iso(),
        "form_stats": dict(form_counts),
    }
    if include_prove:
        payload["prove_stats"] = dict(prove_counts)
    return payload


def fdg_stage2_result_facts(record: RecordState) -> List[JsonDict]:
    result: List[JsonDict] = []
    for fact_id in _ordered_fact_ids(record):
        fact = record["facts"][fact_id]
        result.append(
            {
                "fact_id": fact["fact_id"],
                "text": fact.get("text", ""),
                "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
                "is_final_answer": bool(fact.get("is_final_answer", False)),
                "origin": fact.get("origin", ""),
                "skip": int(fact.get("skip", 1)),
                "proof_obligation": fact.get("proof_obligation") or {},
                "form_status": fact["form_status"],
                "formalization": fact.get("formalization") or fdg_empty_formalization(),
            }
        )
    return result


def fdg_stage3_result_facts(record: RecordState) -> List[JsonDict]:
    result: List[JsonDict] = []
    for fact_id in _ordered_fact_ids(record):
        fact = record["facts"][fact_id]
        result.append(
            {
                "fact_id": fact["fact_id"],
                "text": fact.get("text", ""),
                "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
                "is_final_answer": bool(fact.get("is_final_answer", False)),
                "origin": fact.get("origin", ""),
                "skip": int(fact.get("skip", 1)),
                "proof_obligation": fact.get("proof_obligation") or {},
                "form_status": fact["form_status"],
                "formalization": fact.get("formalization") or fdg_empty_formalization(),
                "prove_status": fact.get("prove_status", "skipped"),
                "solved_lemma": fact.get("solved_lemma") or fdg_empty_solver(skipped=True),
            }
        )
    return result


def fdg_stage2_checkpoint_payload(record: RecordState) -> JsonDict:
    return {
        "meta": record["meta"],
        "input": record["input"],
        "graph": record["graph"],
        "execution": {
            "record_status": record["record_status"],
            "created_at": record["created_at"],
            "updated_at": utc_now_iso(),
        },
        "runtime_facts": {
            fact_id: {
                "fact_id": fact["fact_id"],
                "text": fact.get("text", ""),
                "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
                "is_final_answer": bool(fact.get("is_final_answer", False)),
                "origin": fact.get("origin", ""),
                "skip": int(fact.get("skip", 1)),
                "proof_obligation": fact.get("proof_obligation") or {},
                "form_status": fact["form_status"],
                "form_retries_used": int(fact.get("form_retries_used", 0)),
                "formalization": fact.get("formalization") or fdg_empty_formalization(),
                "form_messages": fact.get("form_messages") or [],
            }
            for fact_id, fact in record["facts"].items()
        },
    }


def fdg_stage3_checkpoint_payload(record: RecordState) -> JsonDict:
    return {
        "meta": record["meta"],
        "input": record["input"],
        "graph": record["graph"],
        "execution": {
            "record_status": record["record_status"],
            "created_at": record["created_at"],
            "updated_at": utc_now_iso(),
        },
        "runtime_facts": {
            fact_id: {
                "fact_id": fact["fact_id"],
                "text": fact.get("text", ""),
                "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
                "is_final_answer": bool(fact.get("is_final_answer", False)),
                "origin": fact.get("origin", ""),
                "skip": int(fact.get("skip", 1)),
                "proof_obligation": fact.get("proof_obligation") or {},
                "form_status": fact["form_status"],
                "formalization": fact.get("formalization") or fdg_empty_formalization(),
                "prove_status": fact.get("prove_status", "skipped"),
                "prove_retries_used": int(fact.get("prove_retries_used", 0)),
                "prove_messages": fact.get("prove_messages") or [],
                "prove_messages_raw": fact.get("prove_messages_raw") or [],
                "solved_lemma": fact.get("solved_lemma") or fdg_empty_solver(skipped=True),
            }
            for fact_id, fact in record["facts"].items()
        },
    }


def fdg_stage2_final_payload(record: RecordState) -> JsonDict:
    meta = dict(record["meta"])
    meta["schema_version"] = "fdg-form-v1"
    meta["stage2_created_at"] = record["created_at"]
    meta["stage2_updated_at"] = utc_now_iso()
    return {
        "meta": meta,
        "input": record["input"],
        "graph": record["graph"],
        "execution": fdg_record_summary(record, include_prove=False),
        "results": {"facts": fdg_stage2_result_facts(record)},
    }


def fdg_stage3_final_payload(record: RecordState) -> JsonDict:
    meta = dict(record["meta"])
    meta["schema_version"] = "fdg-formprove-v1"
    meta["stage3_created_at"] = record["created_at"]
    meta["stage3_updated_at"] = utc_now_iso()
    return {
        "meta": meta,
        "input": record["input"],
        "graph": record["graph"],
        "execution": fdg_record_summary(record, include_prove=True),
        "results": {"facts": fdg_stage3_result_facts(record)},
    }


def fresh_fdg_stage2_record_state(raw: JsonDict) -> RecordState:
    meta = dict(raw.get("meta") or {})
    meta.setdefault("graph_mode", "fdg")
    graph = dict(raw.get("graph") or {})
    facts_raw = list(graph.get("facts") or [])
    document = _fdg_document_from_stage1(raw)

    facts: Dict[str, FactState] = {}
    for fact in document.facts:
        state: FactState = fact.model_dump()
        state["skip"] = _coerce_skip(state.get("skip"), default=1)
        should_execute = fdg_fact_should_execute(state)
        state["proof_obligation"] = build_proof_obligation_from_fact(document, fact.fact_id) if should_execute else {}
        state["form_retries_used"] = 0
        state["form_messages"] = []
        state["formalization"] = fdg_empty_formalization(skipped=not should_execute)
        state["_form_enqueued"] = False
        state["form_status"] = "pending" if should_execute else "skipped"
        facts[fact.fact_id] = state

    if "topo_order" not in graph:
        graph["topo_order"] = [fact["fact_id"] for fact in facts_raw]

    return {
        "meta": meta,
        "input": dict(raw.get("input") or {}),
        "graph": graph,
        "facts": facts,
        "created_at": utc_now_iso(),
        "record_status": "running",
    }


def restore_fdg_stage2_record_state(raw: JsonDict) -> RecordState:
    record = {
        "meta": dict(raw.get("meta") or {}),
        "input": dict(raw.get("input") or {}),
        "graph": dict(raw.get("graph") or {}),
        "facts": {},
        "created_at": raw.get("execution", {}).get("created_at") or utc_now_iso(),
        "record_status": raw.get("execution", {}).get("record_status") or "running",
    }
    for fact_id, fact in (raw.get("runtime_facts") or {}).items():
        restored = dict(fact)
        if restored.get("form_status") == "running":
            restored["form_status"] = "pending"
        restored["_form_enqueued"] = False
        record["facts"][fact_id] = restored
    _ensure_skip_flags(record["facts"])
    for fact in record["facts"].values():
        if not fdg_fact_should_execute(fact):
            fact["form_status"] = "skipped"
            fact["formalization"] = fdg_empty_formalization(skipped=True)
    return record


def fresh_fdg_stage3_record_state(raw: JsonDict) -> RecordState:
    meta = dict(raw.get("meta") or {})
    meta.setdefault("graph_mode", "fdg")
    graph = dict(raw.get("graph") or {})
    result_facts = list((raw.get("results") or {}).get("facts") or [])

    facts: Dict[str, FactState] = {}
    for fact in result_facts:
        state = dict(fact)
        state["skip"] = _coerce_skip(state.get("skip"), default=1)
        state["prove_retries_used"] = 0
        state["prove_messages"] = []
        state["prove_messages_raw"] = []
        state["_prove_enqueued"] = False
        formalization = state.get("formalization") or {}
        if not fdg_fact_should_execute(state):
            state["prove_status"] = "skipped"
            state["solved_lemma"] = fdg_empty_solver(skipped=True)
        elif formalization.get("lean_pass") and formalization.get("lean_code"):
            state["prove_status"] = "pending"
            state["solved_lemma"] = fdg_empty_solver()
        else:
            state["prove_status"] = "skipped"
            state["solved_lemma"] = fdg_empty_solver(skipped=True)
        facts[state["fact_id"]] = state

    return {
        "meta": meta,
        "input": dict(raw.get("input") or {}),
        "graph": graph,
        "facts": facts,
        "created_at": utc_now_iso(),
        "record_status": "running",
    }


def restore_fdg_stage3_record_state(raw: JsonDict) -> RecordState:
    record = {
        "meta": dict(raw.get("meta") or {}),
        "input": dict(raw.get("input") or {}),
        "graph": dict(raw.get("graph") or {}),
        "facts": {},
        "created_at": raw.get("execution", {}).get("created_at") or utc_now_iso(),
        "record_status": raw.get("execution", {}).get("record_status") or "running",
    }
    for fact_id, fact in (raw.get("runtime_facts") or {}).items():
        restored = dict(fact)
        if restored.get("prove_status") == "running":
            restored["prove_status"] = "pending"
        restored["_prove_enqueued"] = False
        record["facts"][fact_id] = restored
    _ensure_skip_flags(record["facts"])
    for fact in record["facts"].values():
        formalization = fact.get("formalization") or {}
        if not fdg_fact_should_execute(fact) or not (formalization.get("lean_pass") and formalization.get("lean_code")):
            fact["prove_status"] = "skipped"
            fact["solved_lemma"] = fdg_empty_solver(skipped=True)
    return record
