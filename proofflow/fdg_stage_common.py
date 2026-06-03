from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .fdg_graph import FDGDocument, build_proof_obligation_from_fact
from .pipeline.context_builders import PARENT_ONLY, build_formalizer_context
from .prompt_builder import build_chat_messages
from .runtime_common import utc_now_iso


JsonDict = Dict[str, Any]
FactState = Dict[str, Any]
RecordState = Dict[str, Any]

FORM_TERMINAL = {"success", "failed", "skipped"}
PROVE_TERMINAL = {"success", "failed", "skipped"}


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


def build_fdg_form_messages(
    fact: FactState,
    *,
    record: RecordState | None = None,
    prompt_name: str = "formalize_obligation",
    context_mode: str = PARENT_ONLY,
) -> List[Dict[str, str]]:
    proof_obligation = fact.get("proof_obligation") or {}
    problem_name = str(proof_obligation.get("problem_name") or f"prove_{fact['fact_id']}").strip()
    lemma_keyword = "theorem" if fact.get("is_final_answer") else "lemma"
    informal_statement = str(proof_obligation.get("informal_statement_content", "")).strip()
    if record is None:
        formalizer_context = "\n\n".join(
            [
                "Current node:\n"
                + str(
                    {
                        "fact_id": fact.get("fact_id", ""),
                        "text": fact.get("text", ""),
                        "parent_fact_ids": list(fact.get("parent_fact_ids") or []),
                        "origin": fact.get("origin", ""),
                        "is_final_answer": bool(fact.get("is_final_answer", False)),
                    }
                ),
                "Target natural language statement:\n" + informal_statement,
            ]
        )
    else:
        formalizer_context = build_formalizer_context(
            record,
            str(fact["fact_id"]),
            mode=context_mode,
        )
    return build_chat_messages(
        "formalize_obligation",
        prompt_name=prompt_name,
        lemma_header=f"{lemma_keyword} {problem_name}",
        paper_theorem_name="test",
        informal_statement_content=informal_statement,
        formalizer_context=formalizer_context,
        formalizer_context_mode=context_mode,
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
