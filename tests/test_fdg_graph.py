from __future__ import annotations

from proofflow.fdg_graph import build_proof_obligation_from_fact, validate_fdg
from proofflow.graph_mode import FDG_GRAPH_MODE, LEGACY_GRAPH_MODE, record_graph_mode


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


def test_validate_fdg_accepts_alpha_example() -> None:
    report = validate_fdg(_alpha_fdg())
    assert report["passed"] is True
    assert report["errors"] == []


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


def test_record_graph_mode_detects_both_modes() -> None:
    legacy_record = {"meta": {"graph_mode": LEGACY_GRAPH_MODE}, "graph": {"nodes": []}}
    fdg_record = {"meta": {"graph_mode": FDG_GRAPH_MODE}, "graph": {"facts": []}}
    assert record_graph_mode(legacy_record) == LEGACY_GRAPH_MODE
    assert record_graph_mode(fdg_record) == FDG_GRAPH_MODE
