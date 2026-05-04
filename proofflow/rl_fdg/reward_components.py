from __future__ import annotations

from typing import Iterable

from .reward_types import RewardWeights


def normalized_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def score_length_penalty(num_facts: int, *, weights: RewardWeights) -> float:
    return 0.0


def invalid_fdg_penalty_scale(*, errors: Iterable[dict], num_facts: int) -> float:
    """Scale graph validation penalty by the fraction of facts with validation errors."""
    if num_facts <= 0:
        return 1.0

    invalid_fact_ids = {
        str(error.get("fact_id", "")).strip()
        for error in errors
        if str(error.get("fact_id", "")).strip()
    }
    if invalid_fact_ids:
        return min(1.0, len(invalid_fact_ids) / float(num_facts))

    has_global_error = any(True for _ in errors)
    return 1.0 if has_global_error else 0.0


def score_structure(
    *,
    valid_json: bool,
    validator_passed: bool,
    warning_count: int,
    num_facts: int,
    errors: Iterable[dict] = (),
    weights: RewardWeights,
) -> tuple[float, float]:
    if not valid_json:
        return weights.invalid_json_penalty, 0.0
    if not validator_passed:
        penalty_scale = invalid_fdg_penalty_scale(errors=errors, num_facts=num_facts)
        return weights.valid_json_bonus + weights.invalid_fdg_penalty * penalty_scale, 0.0

    warning_penalty = warning_count * weights.warning_penalty
    length_penalty = score_length_penalty(num_facts, weights=weights)
    score = weights.valid_json_bonus + weights.validator_pass_bonus - warning_penalty
    return score, length_penalty


def score_formalizer_pass(*, num_formalized: int, num_non_root_facts: int, weights: RewardWeights) -> float:
    return weights.formalizer_pass_weight * normalized_rate(num_formalized, num_non_root_facts)


def score_prover_pass(*, num_proved: int, num_non_root_facts: int, weights: RewardWeights) -> float:
    return weights.prover_pass_weight * normalized_rate(num_proved, num_non_root_facts)


def score_final_answers(*, num_final_verified: int, num_final_facts: int, weights: RewardWeights) -> float:
    return weights.final_answer_pass_weight * normalized_rate(num_final_verified, num_final_facts)


def count_truthy(items: Iterable[bool]) -> int:
    return sum(1 for item in items if item)
