from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from proofflow.fdg_stage2_runner import FDGStage2Runner, build_arg_parser
from proofflow.fdg_stage_common import (
    build_fdg_form_messages,
    build_fdg_form_statement,
    fresh_fdg_stage2_record_state,
)
from proofflow.prompt_builder import build_chat_messages


def _raw_stage1_record() -> dict:
    return {
        "meta": {"record_id": "record_1"},
        "input": {
            "problem": "Let x be a real number. Prove the requested result.",
            "raw_cot": (
                "Public reasoning before.\n"
                "<think>closed secret reasoning</think>\n"
                "Public reasoning after.\n"
                "<think>unfinished secret reasoning"
            ),
        },
        "graph": {
            "topo_order": ["f_root", "f_current", "f_future"],
            "facts": [
                {
                    "fact_id": "f_root",
                    "text": "x is a real number",
                    "parent_fact_ids": [],
                    "origin": "given",
                    "is_final_answer": False,
                    "skip": 1,
                },
                {
                    "fact_id": "f_current",
                    "text": "x + 0 = x",
                    "parent_fact_ids": ["f_root"],
                    "origin": "derived",
                    "is_final_answer": False,
                    "skip": 0,
                },
                {
                    "fact_id": "f_future",
                    "text": "The requested result follows",
                    "parent_fact_ids": ["f_current"],
                    "origin": "answer",
                    "is_final_answer": True,
                    "skip": 0,
                },
            ],
        },
    }


def _record_and_fact() -> tuple[dict, dict]:
    record = fresh_fdg_stage2_record_state(_raw_stage1_record())
    return record, record["facts"]["f_current"]


def _runner_args(context_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        out=Path("stage2_results.jsonl"),
        failed=Path("stage2_failed.jsonl"),
        checkpoint_dir=Path("stage2_ckpt"),
        lean_check_concurrency=1,
        max_pending_validation_batches=1,
        form_batch_size=1,
        formalizer_prompt="formalize_obligation.paper_goedel_v2",
        formalizer_context_mode=context_mode,
    )


def test_c0_messages_match_previous_baseline() -> None:
    record, fact = _record_and_fact()
    proof_obligation = fact["proof_obligation"]
    expected = build_chat_messages(
        "formalize_obligation",
        prompt_name="formalize_obligation.paper_goedel_v2",
        lemma_header="lemma prove_f_current",
        paper_theorem_name="test",
        informal_statement_content=proof_obligation["informal_statement_content"],
    )

    actual = build_fdg_form_messages(
        fact,
        record=record,
        context_mode="c0_parent",
        prompt_name="formalize_obligation.paper_goedel_v2",
    )

    assert actual == expected
    assert record["input"]["problem"] not in actual[-1]["content"]
    assert "f_future" not in actual[-1]["content"]


def test_c1_adds_problem_without_graph_or_cot() -> None:
    record, fact = _record_and_fact()

    statement = build_fdg_form_statement(
        fact,
        record=record,
        context_mode="c1_problem_parent",
    )

    assert record["input"]["problem"] in statement
    assert fact["proof_obligation"]["informal_statement_content"] in statement
    assert "f_root" not in statement
    assert "Public reasoning" not in statement


def test_c2_uses_topological_prefix_and_semantic_fields_only() -> None:
    record, fact = _record_and_fact()

    statement = build_fdg_form_statement(
        fact,
        record=record,
        context_mode="c2_problem_prefix",
    )

    assert '"fact_id": "f_root"' in statement
    assert '"fact_id": "f_current"' in statement
    assert '"fact_id": "f_future"' not in statement
    assert '"skip"' not in statement
    assert "form_status" not in statement
    assert "formalization" not in statement


def test_c3_contains_full_graph_and_current_node_id() -> None:
    record, fact = _record_and_fact()

    statement = build_fdg_form_statement(
        fact,
        record=record,
        context_mode="c3_problem_full_graph",
    )

    assert '"fact_id": "f_root"' in statement
    assert '"fact_id": "f_current"' in statement
    assert '"fact_id": "f_future"' in statement
    assert "Current node id:\nf_current" in statement
    assert '"skip"' not in statement


def test_c4_removes_closed_and_unclosed_think_blocks_without_mutating_raw_cot() -> None:
    record, fact = _record_and_fact()
    original_raw_cot = record["input"]["raw_cot"]

    statement = build_fdg_form_statement(
        fact,
        record=record,
        context_mode="c4_problem_cot_full_graph",
    )

    assert "Public reasoning before." in statement
    assert "Public reasoning after." in statement
    assert "<think>" not in statement
    assert "closed secret reasoning" not in statement
    assert "unfinished secret reasoning" not in statement
    assert record["input"]["raw_cot"] == original_raw_cot


def test_c4_allows_empty_visible_cot_section() -> None:
    record, fact = _record_and_fact()
    record["input"]["raw_cot"] = "<think>secret only</think>"

    statement = build_fdg_form_statement(
        fact,
        record=record,
        context_mode="c4_problem_cot_full_graph",
    )

    assert "Visible solution or chain of thought:\n\nFull graph:" in statement
    assert "secret only" not in statement


def test_invalid_context_mode_is_rejected() -> None:
    record, fact = _record_and_fact()

    with pytest.raises(ValueError, match="Unknown formalizer_context_mode"):
        build_fdg_form_statement(fact, record=record, context_mode="unknown")


def test_non_c0_context_requires_record() -> None:
    _, fact = _record_and_fact()

    with pytest.raises(ValueError, match="requires the full record"):
        build_fdg_form_statement(fact, context_mode="c1_problem_parent")


def test_old_result_without_context_mode_is_only_compatible_with_c0() -> None:
    old_meta = {
        "record_id": "record_1",
        "formalizer_prompt": "formalize_obligation.paper_goedel_v2",
    }
    c0_runner = FDGStage2Runner(_runner_args("c0_parent"))
    c0_runner._validate_existing_prompt_meta(
        old_meta,
        record_id="record_1",
        source="completed result",
    )

    c1_runner = FDGStage2Runner(_runner_args("c1_problem_parent"))
    with pytest.raises(RuntimeError, match="only compatible with c0_parent"):
        c1_runner._validate_existing_prompt_meta(
            old_meta,
            record_id="record_1",
            source="completed result",
        )


def test_stage2_cli_accepts_formalizer_context_mode() -> None:
    args = build_arg_parser().parse_args(
        ["--formalizer-context-mode", "c4_problem_cot_full_graph"]
    )

    assert args.formalizer_context_mode == "c4_problem_cot_full_graph"
