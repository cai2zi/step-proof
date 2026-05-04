from proofflow.rl_fdg.reward_components import (
    invalid_fdg_penalty_scale,
    normalized_rate,
    score_final_answers,
    score_formalizer_pass,
    score_length_penalty,
    score_prover_pass,
    score_structure,
)
from proofflow.rl_fdg.reward_types import RewardWeights


def test_score_structure_for_valid_graph() -> None:
    weights = RewardWeights()
    structure_score, length_penalty = score_structure(
        valid_json=True,
        validator_passed=True,
        warning_count=2,
        num_facts=14,
        weights=weights,
    )
    assert structure_score == weights.valid_json_bonus + weights.validator_pass_bonus - 2 * weights.warning_penalty
    assert length_penalty == 0.0


def test_invalid_json_gets_direct_penalty() -> None:
    weights = RewardWeights()
    structure_score, length_penalty = score_structure(
        valid_json=False,
        validator_passed=False,
        warning_count=0,
        num_facts=0,
        weights=weights,
    )
    assert structure_score == weights.invalid_json_penalty
    assert length_penalty == 0.0


def test_invalid_fdg_penalty_is_weighted_by_invalid_facts() -> None:
    weights = RewardWeights()
    errors = [{"type": "empty_text", "fact_id": "f_2"}, {"type": "invalid_origin", "fact_id": "f_4"}]
    structure_score, length_penalty = score_structure(
        valid_json=True,
        validator_passed=False,
        warning_count=0,
        num_facts=4,
        errors=errors,
        weights=weights,
    )
    assert invalid_fdg_penalty_scale(errors=errors, num_facts=4) == 0.5
    assert structure_score == weights.valid_json_bonus + 0.5 * weights.invalid_fdg_penalty
    assert length_penalty == 0.0


def test_global_invalid_fdg_error_keeps_full_penalty() -> None:
    weights = RewardWeights()
    structure_score, _ = score_structure(
        valid_json=True,
        validator_passed=False,
        warning_count=0,
        num_facts=4,
        errors=[{"type": "missing_final_answer"}],
        weights=weights,
    )
    assert structure_score == weights.valid_json_bonus + weights.invalid_fdg_penalty


def test_pass_rate_scores_are_normalized() -> None:
    weights = RewardWeights()
    assert normalized_rate(1, 4) == 0.25
    assert score_formalizer_pass(num_formalized=2, num_non_root_facts=4, weights=weights) == 0.2
    assert score_prover_pass(num_proved=1, num_non_root_facts=4, weights=weights) == 0.05
    assert score_final_answers(num_final_verified=1, num_final_facts=2, weights=weights) == 0.1
    assert score_length_penalty(12, weights=weights) == 0.0
    assert score_length_penalty(100, weights=weights) == 0.0
