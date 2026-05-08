from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json5
from pydantic import BaseModel, Field, ValidationError

from .prompt_builder import build_chat_messages


LEGACY_FDG_ORIGINS = {
    "problem",
    "definition",
    "derived",
    "theorem",
    "assumption",
    "approximation",
    "other",
}
ORIGIN4_FDG_ORIGINS = {
    "given",
    "introduced",
    "derived",
    "answer",
}
FDG_ORIGINS = LEGACY_FDG_ORIGINS
FDG_ORIGIN_SCHEMAS = {
    "legacy": LEGACY_FDG_ORIGINS,
    "origin4": ORIGIN4_FDG_ORIGINS,
}
FDG_PROMPT_ORIGIN_SCHEMAS = {
    "fdg": "legacy",
    "fdg_v1": "legacy",
    "fdg_origin4": "origin4",
}

JsonDict = Dict[str, Any]


def _normalize_prompt_name(prompt_name: str | None) -> str:
    name = str(prompt_name or "fdg").strip() or "fdg"
    return name[:-3] if name.endswith(".md") else name


def fdg_origin_schema_for_prompt(prompt_name: str | None) -> str:
    name = _normalize_prompt_name(prompt_name)
    return FDG_PROMPT_ORIGIN_SCHEMAS.get(name, "legacy")


def fdg_allowed_origins_for_prompt(prompt_name: str | None) -> set[str]:
    return set(FDG_ORIGIN_SCHEMAS[fdg_origin_schema_for_prompt(prompt_name)])


class FactItem(BaseModel):
    fact_id: str = Field(..., description="Fact identifier like f_1")
    text: str = Field(..., description="Atomic mathematical fact")
    parent_fact_ids: List[str] = Field(default_factory=list)
    is_final_answer: bool = Field(default=False)
    origin: str = Field(..., description="Metadata origin")


class FDGDocument(BaseModel):
    problem_id: str = Field(..., description="Problem identifier")
    problem_text: str = Field(..., description="Problem statement")
    facts: List[FactItem] = Field(default_factory=list)


@dataclass
class FDGParseResult:
    ok: bool
    document: Optional[FDGDocument]
    error_msg: Optional[str]
    report: JsonDict


def strip_think_blocks(text: str) -> str:
    without_closed = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think\b[^>]*>.*\Z", "", without_closed, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_json_block(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    else:
        text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if text_no_think.strip():
            text = text_no_think

    patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            for candidate in reversed(matches):
                candidate = candidate.strip()
                if candidate.startswith("{"):
                    return candidate
            return matches[-1].strip()

    def find_balanced(source: str, open_char: str, close_char: str) -> Optional[str]:
        depth = 0
        start_idx = -1
        for index, char in enumerate(source):
            if char == open_char:
                if start_idx == -1:
                    start_idx = index
                depth += 1
            elif char == close_char and start_idx != -1:
                depth -= 1
                if depth == 0:
                    return source[start_idx:index + 1]
        return None

    obj_match = find_balanced(text, "{", "}")
    if obj_match:
        return obj_match
    raise ValueError(f"No JSON block found. Text starts with: {repr(text[:100])}")


def _sanitize_backslashes(raw: str) -> str:
    raw = raw.replace("\\\\", "\\")
    raw = raw.replace("\\in", "\\\\in")
    raw = raw.replace("\\Q", "\\\\Q")
    raw = raw.replace(r"\int", r"\\int")
    raw = raw.replace(r"\mathbb{Q}", r"\\mathbb{Q}")
    return raw.replace("\\", "\\\\")


def parse_llm_json(content: str) -> Any:
    raw = _extract_json_block(content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sanitized = _sanitize_backslashes(raw)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            return json5.loads(sanitized)


def _report_entry(
    kind: str,
    message: str,
    *,
    fact_id: str | None = None,
) -> JsonDict:
    payload: JsonDict = {"type": kind, "message": message}
    if fact_id:
        payload["fact_id"] = fact_id
    return payload


def _looks_non_atomic(text: str) -> bool:
    lowered = f" {text.lower()} "
    connectors = [" and ", " but ", ";", "；", " 并且", " 且", " 同时"]
    indicator_count = sum(1 for marker in connectors if marker in lowered)
    math_ops = len(re.findall(r"(=|<=|>=|<|>|\\le|\\ge)", text))
    return indicator_count > 0 and math_ops >= 2


def _contains_narrative_words(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ["therefore", "hence", "thus", "next", "then we", "we compute"])


def _looks_like_case_assumption(text: str) -> bool:
    lowered = text.lower().strip()
    if lowered.startswith("if ") or lowered.startswith("under the assumption"):
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9_α-ωΑ-Ω\-\+\*/\^\(\)\| ]*(<|>|<=|>=|=)[a-zA-Z0-9_α-ωΑ-Ω\-\+\*/\^\(\)\| ]+", text.strip()))


def _join_parent_facts(parents: List[str]) -> str:
    if not parents:
        return ""
    if len(parents) == 1:
        return parents[0]
    if len(parents) == 2:
        return f"{parents[0]} and {parents[1]}"
    return ", ".join(parents[:-1]) + f", and {parents[-1]}"


def fdg_topo_order(facts: List[FactItem | JsonDict]) -> List[str]:
    return [str((fact.fact_id if isinstance(fact, FactItem) else fact.get("fact_id"))).strip() for fact in facts]


def fdg_final_fact_ids(facts: List[FactItem | JsonDict]) -> List[str]:
    final_ids: List[str] = []
    for fact in facts:
        if isinstance(fact, FactItem):
            if fact.is_final_answer:
                final_ids.append(fact.fact_id)
        elif fact.get("is_final_answer"):
            final_ids.append(str(fact.get("fact_id", "")).strip())
    return final_ids


def validate_fdg(fdg: JsonDict | FDGDocument, *, prompt_name: str = "fdg") -> JsonDict:
    origin_schema = fdg_origin_schema_for_prompt(prompt_name)
    allowed_origins = fdg_allowed_origins_for_prompt(prompt_name)
    report: JsonDict = {
        "passed": False,
        "errors": [],
        "warnings": [],
        "origin_schema": origin_schema,
    }
    errors: List[JsonDict] = report["errors"]
    warnings: List[JsonDict] = report["warnings"]

    try:
        document = fdg if isinstance(fdg, FDGDocument) else FDGDocument.model_validate(fdg)
    except ValidationError as exc:
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", []))
            errors.append(
                _report_entry(
                    "schema_validation_error",
                    f"{location}: {err.get('msg', 'invalid value')}",
                )
            )
        return report

    if not document.facts:
        errors.append(_report_entry("empty_facts", "The FDG must contain at least one fact."))
        return report

    seen_ids: set[str] = set()
    id_to_index: Dict[str, int] = {}
    id_to_fact: Dict[str, FactItem] = {}
    for index, fact in enumerate(document.facts):
        fact_id = fact.fact_id.strip()
        text = fact.text.strip()
        origin = fact.origin.strip().lower()

        if not fact_id:
            errors.append(_report_entry("empty_fact_id", "A fact has an empty fact_id."))
            continue
        if fact_id in seen_ids:
            errors.append(
                _report_entry(
                    "duplicate_fact_id",
                    f"Fact id {fact_id} appears more than once.",
                    fact_id=fact_id,
                )
            )
        seen_ids.add(fact_id)
        id_to_index[fact_id] = index
        id_to_fact[fact_id] = fact

        if not text:
            errors.append(_report_entry("empty_text", "Fact text must not be empty.", fact_id=fact_id))

        if origin not in allowed_origins:
            errors.append(
                _report_entry(
                    "invalid_origin",
                    f"Origin {fact.origin!r} is not one of {sorted(allowed_origins)}.",
                    fact_id=fact_id,
                )
            )

        if len(set(fact.parent_fact_ids)) != len(fact.parent_fact_ids):
            errors.append(
                _report_entry(
                    "duplicate_parent_fact",
                    "parent_fact_ids contains duplicates.",
                    fact_id=fact_id,
                )
            )

        if _looks_non_atomic(text):
            warnings.append(
                _report_entry(
                    "possibly_non_atomic_fact",
                    "This fact appears to contain multiple assertions.",
                    fact_id=fact_id,
                )
            )
        if _contains_narrative_words(text):
            warnings.append(
                _report_entry(
                    "narrative_fact_text",
                    "This fact contains narrative wording and may not be purely factual.",
                    fact_id=fact_id,
                )
            )

    final_ids = [fact.fact_id for fact in document.facts if fact.is_final_answer]
    if not final_ids:
        errors.append(_report_entry("missing_final_answer", "At least one fact must have is_final_answer=true."))
    if origin_schema == "origin4" and len(final_ids) > 1:
        errors.append(
            _report_entry(
                "multiple_final_answers",
                "The fdg_origin4 schema requires exactly one fact with is_final_answer=true.",
            )
        )

    adjacency: Dict[str, List[str]] = {fact.fact_id: [] for fact in document.facts}
    reverse_adjacency: Dict[str, List[str]] = {fact.fact_id: [] for fact in document.facts}
    for index, fact in enumerate(document.facts):
        fact_id = fact.fact_id
        parent_ids = [parent_id.strip() for parent_id in fact.parent_fact_ids if parent_id.strip()]
        for parent_id in parent_ids:
            if parent_id not in id_to_fact:
                errors.append(
                    _report_entry(
                        "missing_parent_fact",
                        f"Parent fact {parent_id} does not exist.",
                        fact_id=fact_id,
                    )
                )
                continue
            if id_to_index[parent_id] >= index:
                errors.append(
                    _report_entry(
                        "forward_parent_fact",
                        f"Parent fact {parent_id} must appear before {fact_id}.",
                        fact_id=fact_id,
                    )
                )
            adjacency[parent_id].append(fact_id)
            reverse_adjacency[fact_id].append(parent_id)

        origin = fact.origin.strip().lower()
        if origin_schema == "origin4":
            if origin == "given" and parent_ids:
                errors.append(
                    _report_entry(
                        "given_fact_with_parents",
                        "A given fact must not have parent_fact_ids.",
                        fact_id=fact_id,
                    )
                )
            if origin == "derived" and not parent_ids:
                errors.append(
                    _report_entry(
                        "derived_without_parents",
                        "A derived fact must list at least one parent_fact_ids entry.",
                        fact_id=fact_id,
                    )
                )
            if origin == "answer" and not fact.is_final_answer:
                errors.append(
                    _report_entry(
                        "answer_origin_without_final_flag",
                        'A fact with origin="answer" must have is_final_answer=true.',
                        fact_id=fact_id,
                    )
                )
            if fact.is_final_answer and origin != "answer":
                errors.append(
                    _report_entry(
                        "final_answer_origin_mismatch",
                        'A final-answer fact must have origin="answer" under fdg_origin4.',
                        fact_id=fact_id,
                    )
                )
        else:
            if origin == "derived" and not parent_ids:
                errors.append(
                    _report_entry(
                        "derived_without_parents",
                        "A derived fact must list at least one parent_fact_ids entry.",
                        fact_id=fact_id,
                    )
                )
            if origin == "problem" and parent_ids:
                errors.append(
                    _report_entry(
                        "problem_fact_with_parents",
                        "A problem-origin fact should not have parent_fact_ids.",
                        fact_id=fact_id,
                    )
                )
            if origin == "assumption" and not parent_ids and _looks_like_case_assumption(fact.text):
                warnings.append(
                    _report_entry(
                        "possible_globalized_case_assumption",
                        "This assumption looks like a local case assumption promoted to a root fact.",
                        fact_id=fact_id,
                    )
                )
        if fact.is_final_answer and not parent_ids and fact.text.strip() not in document.problem_text:
            warnings.append(
                _report_entry(
                    "final_answer_without_parents",
                    "This final-answer fact has no parents and does not appear verbatim in the problem.",
                    fact_id=fact_id,
                )
            )

    visit_state: Dict[str, int] = {fact_id: 0 for fact_id in adjacency}

    def has_cycle(node_id: str) -> bool:
        if visit_state[node_id] == 1:
            return True
        if visit_state[node_id] == 2:
            return False
        visit_state[node_id] = 1
        for child_id in adjacency.get(node_id, []):
            if has_cycle(child_id):
                return True
        visit_state[node_id] = 2
        return False

    for fact_id in adjacency:
        if visit_state[fact_id] == 0 and has_cycle(fact_id):
            errors.append(_report_entry("cycle_detected", "The FDG contains a cycle.", fact_id=fact_id))
            break

    reachable_to_final: set[str] = set()
    stack = list(final_ids)
    while stack:
        current = stack.pop()
        if current in reachable_to_final:
            continue
        reachable_to_final.add(current)
        stack.extend(reverse_adjacency.get(current, []))

    for fact in document.facts:
        if fact.fact_id not in reachable_to_final:
            errors.append(
                _report_entry(
                    "dead_end_fact",
                    "This non-final fact does not contribute to any final answer fact.",
                    fact_id=fact.fact_id,
                )
            )

    report["passed"] = not errors
    return report


def _format_report_for_retry(report: JsonDict) -> str:
    errors = report.get("errors") or []
    if not errors:
        return "FDG validation failed for unspecified reasons. Regenerate the full FDG JSON."
    lines = ["FDG validation failed. Regenerate the entire FDG JSON and fix these errors:"]
    for index, entry in enumerate(errors, start=1):
        fact_id = str(entry.get("fact_id", "")).strip()
        prefix = f"{index}. [{fact_id}] " if fact_id else f"{index}. "
        lines.append(prefix + str(entry.get("message", "unknown error")))
    return "\n".join(lines)


def build_fdg_messages(
    *,
    problem_text: str,
    solution_or_cot: str,
    include_think_in_dag: bool = True,
    prompt_name: str = "fdg",
) -> List[Dict[str, str]]:
    if not problem_text or not solution_or_cot:
        raise ValueError("FDG generation requires non-empty problem_text and solution_or_cot.")
    dag_solution = solution_or_cot if include_think_in_dag else strip_think_blocks(solution_or_cot)
    return build_chat_messages(
        "fdg",
        prompt_name=prompt_name,
        problem_text=problem_text,
        solution_or_cot=dag_solution,
    )


def parse_and_validate_fdg(content: str, *, prompt_name: str = "fdg") -> FDGParseResult:
    try:
        raw_payload = parse_llm_json(content)
    except Exception as exc:
        report = {
            "passed": False,
            "errors": [_report_entry("json_parse_error", f"Failed to parse FDG JSON: {exc}")],
            "warnings": [],
        }
        return FDGParseResult(ok=False, document=None, error_msg=_format_report_for_retry(report), report=report)

    report = validate_fdg(raw_payload, prompt_name=prompt_name)
    if not report["passed"]:
        return FDGParseResult(
            ok=False,
            document=None,
            error_msg=_format_report_for_retry(report),
            report=report,
        )

    document = FDGDocument.model_validate(raw_payload)
    return FDGParseResult(ok=True, document=document, error_msg=None, report=report)


def build_proof_obligation_from_fact(fdg: JsonDict | FDGDocument, fact_id: str) -> JsonDict:
    document = fdg if isinstance(fdg, FDGDocument) else FDGDocument.model_validate(fdg)
    fact_lookup = {fact.fact_id: fact for fact in document.facts}
    current = fact_lookup.get(fact_id)
    if current is None:
        raise KeyError(f"Unknown fact_id: {fact_id}")
    if not current.parent_fact_ids:
        raise ValueError(f"Fact {fact_id} is a root fact and has no proof obligation.")

    parent_texts: List[str] = []
    for parent_id in current.parent_fact_ids:
        parent = fact_lookup.get(parent_id)
        if parent is None:
            raise KeyError(f"Missing parent fact for obligation: {parent_id}")
        parent_texts.append(parent.text.strip())

    statement = f"Given {_join_parent_facts(parent_texts)}, prove that {current.text.strip()}."
    return {
        "problem_name": f"prove_{current.fact_id}",
        "informal_statement_content": statement,
    }
