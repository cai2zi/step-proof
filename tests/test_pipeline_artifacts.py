from __future__ import annotations

import pytest

from proofflow.pipeline.artifacts import (
    ArtifactManager,
    stage_fingerprint,
    stable_fingerprint,
)
from proofflow.pipeline.specs import ModelSpec, StageSpec


def _stage(context_mode: str = "parent_only", temperature: float = 0.0) -> StageSpec:
    return StageSpec(
        name="stage2",
        prompt_name="formalize_obligation.context_ablation",
        context_mode=context_mode,
        model=ModelSpec(
            backend="api",
            name="gpt-5-mini",
            temperature=temperature,
        ),
    )


def test_stable_fingerprint_is_order_independent():
    left = stable_fingerprint({"a": 1, "b": {"c": 2}})
    right = stable_fingerprint({"b": {"c": 2}, "a": 1})

    assert left == right


def test_stage_fingerprint_changes_with_context_mode():
    parent = stage_fingerprint(_stage("parent_only"), schema_version="step-proof-v2")
    full_graph = stage_fingerprint(
        _stage("problem_full_graph"),
        schema_version="step-proof-v2",
    )

    assert parent != full_graph


def test_stage_fingerprint_changes_with_model_temperature():
    cold = stage_fingerprint(_stage(temperature=0.0), schema_version="step-proof-v2")
    warm = stage_fingerprint(_stage(temperature=0.7), schema_version="step-proof-v2")

    assert cold != warm


def test_checkpoint_fingerprint_mismatch_is_rejected():
    checkpoint = {
        "meta": {
            "record_id": "case-1",
            "stage_fingerprint": "old",
        }
    }

    with pytest.raises(RuntimeError, match="stage_fingerprint"):
        ArtifactManager.assert_checkpoint_fingerprint(
            checkpoint,
            expected_fingerprint="new",
            stage_name="stage2",
        )


def test_stamp_meta_sets_v2_schema_and_fingerprint():
    meta = {"record_id": "case-1"}

    ArtifactManager.stamp_meta(meta, stage_name="stage2", fingerprint="abc")

    assert meta["schema_version"] == "step-proof-v2"
    assert meta["stage_name"] == "stage2"
    assert meta["stage_fingerprint"] == "abc"
