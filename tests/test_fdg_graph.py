from __future__ import annotations

import json

from proofflow.fdg_graph import build_proof_obligation_from_fact, parse_and_validate_fdg, validate_fdg
from proofflow.graph_mode import FDG_GRAPH_MODE, record_graph_mode


def _alpha_fdg() -> dict:
    return {
        "problem_id": "alpha_example",
        "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
        "facts": [
            {
                "fact_id": "f_1",
                "text": "sin alpha = 3/5",
                "parent_fact_ids": [],
                "is_final_answer": False,
                "origin": "problem",
            },
            {
                "fact_id": "f_2",
                "text": "-pi/2 < alpha < pi/2",
                "parent_fact_ids": [],
                "is_final_answer": False,
                "origin": "problem",
            },
            {
                "fact_id": "f_3",
                "text": "cos alpha = 4/5",
                "parent_fact_ids": ["f_1", "f_2"],
                "is_final_answer": False,
                "origin": "derived",
            },
            {
                "fact_id": "f_4",
                "text": "cot(2 alpha) = 7/24",
                "parent_fact_ids": ["f_1", "f_3"],
                "is_final_answer": True,
                "origin": "derived",
            },
        ],
    }


def _origin4_fdg() -> dict:
    return {
        "problem_id": "alpha_example",
        "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
        "facts": [
            {
                "fact_id": "f_1",
                "text": "sin alpha = 3/5",
                "parent_fact_ids": [],
                "is_final_answer": False,
                "origin": "given",
            },
            {
                "fact_id": "f_2",
                "text": "-pi/2 < alpha < pi/2",
                "parent_fact_ids": [],
                "is_final_answer": False,
                "origin": "given",
            },
            {
                "fact_id": "f_3",
                "text": "cos alpha = 4/5",
                "parent_fact_ids": ["f_1", "f_2"],
                "is_final_answer": False,
                "origin": "derived",
            },
            {
                "fact_id": "f_4",
                "text": "cot(2 alpha) = 7/24",
                "parent_fact_ids": ["f_1", "f_3"],
                "is_final_answer": True,
                "origin": "answer",
            },
        ],
    }


def test_validate_fdg_accepts_alpha_example() -> None:
    report = validate_fdg(_alpha_fdg())
    assert report["passed"] is True
    assert report["errors"] == []


def test_validate_fdg_origin4_accepts_alpha_example() -> None:
    report = validate_fdg(_origin4_fdg(), prompt_name="fdg_origin4")
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["origin_schema"] == "origin4"


def test_validate_fdg_origin4_rejects_legacy_origins() -> None:
    report = validate_fdg(_alpha_fdg(), prompt_name="fdg_origin4")
    assert report["passed"] is False
    assert any(error["type"] == "invalid_origin" for error in report["errors"])


def test_validate_fdg_origin4_rejects_derived_without_parents() -> None:
    payload = _origin4_fdg()
    payload["facts"][2]["parent_fact_ids"] = []
    report = validate_fdg(payload, prompt_name="fdg_origin4")
    assert report["passed"] is False
    assert any(error["type"] == "derived_without_parents" for error in report["errors"])


def test_validate_fdg_origin4_rejects_given_with_parents() -> None:
    payload = _origin4_fdg()
    payload["facts"][1]["parent_fact_ids"] = ["f_1"]
    report = validate_fdg(payload, prompt_name="fdg_origin4")
    assert report["passed"] is False
    assert any(error["type"] == "given_fact_with_parents" for error in report["errors"])


def test_validate_fdg_origin4_rejects_final_answer_origin_mismatch() -> None:
    payload = _origin4_fdg()
    payload["facts"][-1]["origin"] = "derived"
    report = validate_fdg(payload, prompt_name="fdg_origin4")
    assert report["passed"] is False
    assert any(error["type"] == "final_answer_origin_mismatch" for error in report["errors"])


def test_validate_fdg_origin4_rejects_multiple_final_answers() -> None:
    payload = _origin4_fdg()
    payload["facts"][2]["is_final_answer"] = True
    payload["facts"][2]["origin"] = "answer"
    report = validate_fdg(payload, prompt_name="fdg_origin4")
    assert report["passed"] is False
    assert any(error["type"] == "multiple_final_answers" for error in report["errors"])


def test_parse_and_validate_fdg_origin4_accepts_valid_json() -> None:
    result = parse_and_validate_fdg(json.dumps(_origin4_fdg()), prompt_name="fdg_origin4")
    assert result.ok is True
    assert result.document is not None


def test_validate_fdg_rejects_forward_reference() -> None:
    payload = _alpha_fdg()
    payload["facts"][0]["parent_fact_ids"] = ["f_3"]
    report = validate_fdg(payload)
    assert report["passed"] is False
    assert any(error["type"] == "forward_parent_fact" for error in report["errors"])


def test_validate_fdg_rejects_missing_final_answer() -> None:
    payload = _alpha_fdg()
    payload["facts"][-1]["is_final_answer"] = False
    report = validate_fdg(payload)
    assert report["passed"] is False
    assert any(error["type"] == "missing_final_answer" for error in report["errors"])


def test_validate_fdg_warns_for_narrative_text() -> None:
    payload = _alpha_fdg()
    payload["facts"][2]["text"] = "Therefore, cos alpha = 4/5"
    report = validate_fdg(payload)
    assert report["passed"] is True
    assert any(warning["type"] == "narrative_fact_text" for warning in report["warnings"])


def test_build_proof_obligation_from_fact() -> None:
    obligation = build_proof_obligation_from_fact(_alpha_fdg(), "f_4")
    assert obligation["problem_name"] == "prove_f_4"
    assert (
        obligation["informal_statement_content"]
        == "Given sin alpha = 3/5 and cos alpha = 4/5, prove that cot(2 alpha) = 7/24."
    )


def test_record_graph_mode_detects_fdg_mode() -> None:
    fdg_record = {"meta": {"graph_mode": FDG_GRAPH_MODE}, "graph": {"facts": []}}
    assert record_graph_mode(fdg_record) == FDG_GRAPH_MODE
