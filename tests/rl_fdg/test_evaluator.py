import asyncio

from proofflow.rl_fdg.evaluator import FDGRLEvaluator
from proofflow.rl_fdg.reward_types import (
    BridgeFactResult,
    CandidateGraphInput,
    FDGRLEvaluatorConfig,
    LeanRuntimeConfig,
    ModelRuntimeConfig,
    RewardWeights,
)


class FakeFormalizerBridge:
    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def batch_formalize(self, tasks):
        results = []
        for task in tasks:
            results.append(
                BridgeFactResult(
                    sample_index=task.sample_index,
                    fact_id=task.fact["fact_id"],
                    stage="formalizer",
                    attempts=1,
                    success=True,
                    verified=False,
                    lean_code=f"lemma {task.fact['fact_id']} : True := by trivial",
                    error_message="",
                    raw_output="",
                    attempt_history=[],
                )
            )
        return results


class FakeProverBridge:
    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def batch_prove(self, tasks):
        results = []
        for task in tasks:
            verified = task.fact["fact_id"] == "f_4"
            results.append(
                BridgeFactResult(
                    sample_index=task.sample_index,
                    fact_id=task.fact["fact_id"],
                    stage="prover",
                    attempts=1,
                    success=verified,
                    verified=verified,
                    lean_code=f"lemma {task.fact['fact_id']} : True := by trivial",
                    error_message="" if verified else "failed",
                    raw_output="",
                    attempt_history=[],
                )
            )
        return results


def _config() -> FDGRLEvaluatorConfig:
    model_cfg = ModelRuntimeConfig(
        model_path="dummy",
        gpus="0",
        tensor_parallel_size=1,
        max_tokens=128,
        token_limit=1024,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        seed=42,
        top_k=20,
        retries=1,
        batch_size=8,
        prompt_name="dummy",
    )
    return FDGRLEvaluatorConfig(
        weights=RewardWeights(),
        formalizer=model_cfg,
        prover=model_cfg,
        lean=LeanRuntimeConfig(
            mathlib_path="dummy",
            backend="subprocess",
            check_concurrency=1,
            worker_pool_size=0,
            temp_dir="dummy",
        ),
        include_prover=True,
    )


def test_evaluator_scores_valid_graph_with_prover_signal() -> None:
    output = """
{
  "problem_id": "alpha_example",
  "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
  "facts": [
    {"fact_id": "f_1", "text": "sin alpha = 3/5", "parent_fact_ids": [], "is_final_answer": false, "origin": "problem"},
    {"fact_id": "f_2", "text": "-pi/2 < alpha < pi/2", "parent_fact_ids": [], "is_final_answer": false, "origin": "problem"},
    {"fact_id": "f_3", "text": "cos alpha = 4/5", "parent_fact_ids": ["f_1", "f_2"], "is_final_answer": false, "origin": "derived"},
    {"fact_id": "f_4", "text": "cot(2 alpha) = 7/24", "parent_fact_ids": ["f_1", "f_3"], "is_final_answer": true, "origin": "derived"}
  ]
}
""".strip()
    evaluator = FDGRLEvaluator(
        _config(),
        formalizer_bridge=FakeFormalizerBridge(),
        prover_bridge=FakeProverBridge(),
        lean_server=object(),  # sentinel to avoid creating a real Lean runtime
        owned_lean_server=False,
    )
    result = evaluator.evaluate_batch_sync(
        [
            CandidateGraphInput(
                record_id="alpha_example",
                problem_text="Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
                solution_or_cot="We compute cos alpha then cot(2 alpha).",
                model_output=output,
            )
        ]
    )[0]
    assert result.validator_passed is True
    assert result.num_non_root_facts == 2
    assert result.num_formalized == 2
    assert result.num_proved == 1
    assert result.num_final_verified == 1
    assert result.prover_score > 0
    assert result.final_answer_score > 0
    evaluator.close()


def test_evaluator_penalizes_invalid_json() -> None:
    evaluator = FDGRLEvaluator(
        _config(),
        formalizer_bridge=FakeFormalizerBridge(),
        prover_bridge=FakeProverBridge(),
        lean_server=object(),
        owned_lean_server=False,
    )
    result = evaluator.evaluate_batch_sync(
        [
            CandidateGraphInput(
                record_id="bad_case",
                problem_text="problem",
                solution_or_cot="cot",
                model_output="{not valid json",
            )
        ]
    )[0]
    assert result.valid_json is False
    assert result.score < 0
    evaluator.close()
