from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from proofflow.fdg_graph import FDGDocument, parse_llm_json, validate_fdg


JsonDict = Dict[str, Any]


@dataclass
class ParsedFDGCandidate:
    valid_json: bool
    validator_passed: bool
    raw_payload: Optional[JsonDict]
    document: Optional[FDGDocument]
    report: JsonDict
    parse_error: Optional[str] = None


def parse_fdg_candidate(output_text: str, *, prompt_name: str = "fdg") -> ParsedFDGCandidate:
    try:
        raw_payload = parse_llm_json(output_text)
    except Exception as exc:
        return ParsedFDGCandidate(
            valid_json=False,
            validator_passed=False,
            raw_payload=None,
            document=None,
            report={
                "passed": False,
                "errors": [{"type": "json_parse_error", "message": f"Failed to parse FDG JSON: {exc}"}],
                "warnings": [],
            },
            parse_error=str(exc),
        )

    report = validate_fdg(raw_payload, prompt_name=prompt_name)
    if not report["passed"]:
        return ParsedFDGCandidate(
            valid_json=True,
            validator_passed=False,
            raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
            document=None,
            report=report,
            parse_error=None,
        )

    return ParsedFDGCandidate(
        valid_json=True,
        validator_passed=True,
        raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        document=FDGDocument.model_validate(raw_payload),
        report=report,
        parse_error=None,
    )
