from __future__ import annotations

from proofflow.fdg_stage_common import build_fdg_form_messages
from proofflow.pipeline.context_builders import (
    PARENT_ONLY,
    PROBLEM_COT_FULL_GRAPH,
    PROBLEM_FULL_GRAPH,
    PROBLEM_PARENT,
    PROBLEM_PREFIX,
    build_formalizer_context,
    normalize_context_mode,
)


def _record():
    return {
        "meta": {"record_id": "case-1"},
        "input": {
            "problem": "Let x be a natural number. Prove x + 0 = x.",
            "raw_cot": "We use the identity theorem for addition by zero.",
        },
        "graph": {
            "topo_order": ["f0", "f1", "f2"],
            "facts": [
                {
                    "fact_id": "f0",
                    "text": "x is a natural number",
                    "parent_fact_ids": [],
                    "origin": "problem",
                    "is_final_answer": False,
                    "skip": 1,
                },
                {
                    "fact_id": "f1",
                    "text": "x + 0 = x",
                    "parent_fact_ids": ["f0"],
                    "origin": "solution",
                    "is_final_answer": True,
                    "skip": 0,
                    "proof_obligation": {
                        "problem_name": "prove_f1",
                        "informal_statement_content": (
                            "Given x is a natural number, prove that x + 0 = x."
                        ),
                    },
                },
                {
                    "fact_id": "f2",
                    "text": "unused later node",
                    "parent_fact_ids": ["f1"],
                    "origin": "solution",
                    "is_final_answer": False,
                    "skip": 0,
                },
            ],
        },
    }


def test_context_mode_aliases():
    assert normalize_context_mode("c0_parent") == PARENT_ONLY
    assert normalize_context_mode("c1") == PROBLEM_PARENT
    assert normalize_context_mode("c2_problem_prefix") == PROBLEM_PREFIX
    assert normalize_context_mode("c3") == PROBLEM_FULL_GRAPH
    assert normalize_context_mode("c4") == PROBLEM_COT_FULL_GRAPH


def test_parent_only_context_contains_current_and_parent_but_not_problem():
    context = build_formalizer_context(_record(), "f1", mode=PARENT_ONLY)

    assert "Current node" in context
    assert "Parent node(s)" in context
    assert "x + 0 = x" in context
    assert "x is a natural number" in context
    assert "Problem:" not in context


def test_problem_prefix_context_uses_topological_prefix_only():
    context = build_formalizer_context(_record(), "f1", mode=PROBLEM_PREFIX)

    assert "Problem:" in context
    assert "Graph prefix before current node" in context
    assert "x is a natural number" in context
    assert "unused later node" not in context


def test_problem_full_graph_context_includes_current_node_id():
    context = build_formalizer_context(_record(), "f1", mode=PROBLEM_FULL_GRAPH)

    assert "Full graph" in context
    assert "Current node id" in context
    assert "f1" in context
    assert "unused later node" in context


def test_problem_cot_full_graph_context_includes_cot():
    context = build_formalizer_context(_record(), "f1", mode=PROBLEM_COT_FULL_GRAPH)

    assert "Full CoT" in context
    assert "identity theorem" in context
    assert "Full graph" in context


def test_context_ablation_prompt_renders_formalizer_context():
    record = _record()
    fact = record["graph"]["facts"][1]
    messages = build_fdg_form_messages(
        fact,
        record=record,
        prompt_name="formalize_obligation.context_ablation",
        context_mode=PROBLEM_PARENT,
    )

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "Context mode:" in content
    assert PROBLEM_PARENT in content
    assert "Let x be a natural number" in content
    assert "Given x is a natural number" in content
