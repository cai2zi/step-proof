import asyncio

from proofflow.rl_fdg.evaluator import FDGRLEvaluator
from proofflow.rl_fdg.reward_types import (
    BridgeFactResult,
    CandidateGraphInput,
    FDGRLEvaluatorConfig,
    LeanRuntimeConfig,
    ModelRuntimeConfig,
    RewardWeights,
    SchedulerRuntimeConfig,
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


def _model_config(batch_size: int = 8) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
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
        batch_size=batch_size,
        prompt_name="dummy",
    )


def _config(
    formalizer_batch_size: int = 8,
    prover_batch_size: int = 8,
    fdg_prompt: str = "fdg",
) -> FDGRLEvaluatorConfig:
    return FDGRLEvaluatorConfig(
        weights=RewardWeights(),
        formalizer=_model_config(batch_size=formalizer_batch_size),
        prover=_model_config(batch_size=prover_batch_size),
        lean=LeanRuntimeConfig(
            mathlib_path="dummy",
            backend="subprocess",
            check_concurrency=1,
            worker_pool_size=0,
            temp_dir="dummy",
        ),
        scheduler=SchedulerRuntimeConfig(
            graph_wait_ms=0,
            max_graph_batch_size=64,
            formalizer_wait_ms=0,
            prover_wait_ms=0,
        ),
        include_prover=True,
        fdg_prompt=fdg_prompt,
    )


def _valid_graph_output(record_id: str = "alpha_example") -> str:
    return f"""
{{
  "problem_id": "{record_id}",
  "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
  "facts": [
    {{"fact_id": "f_1", "text": "sin alpha = 3/5", "parent_fact_ids": [], "is_final_answer": false, "origin": "problem"}},
    {{"fact_id": "f_2", "text": "-pi/2 < alpha < pi/2", "parent_fact_ids": [], "is_final_answer": false, "origin": "problem"}},
    {{"fact_id": "f_3", "text": "cos alpha = 4/5", "parent_fact_ids": ["f_1", "f_2"], "is_final_answer": false, "origin": "derived"}},
    {{"fact_id": "f_4", "text": "cot(2 alpha) = 7/24", "parent_fact_ids": ["f_1", "f_3"], "is_final_answer": true, "origin": "derived"}}
  ]
}}
""".strip()


def _valid_origin4_graph_output(record_id: str = "alpha_example") -> str:
    return f"""
{{
  "problem_id": "{record_id}",
  "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5, find cot(2 alpha).",
  "facts": [
    {{"fact_id": "f_1", "text": "sin alpha = 3/5", "parent_fact_ids": [], "is_final_answer": false, "origin": "given"}},
    {{"fact_id": "f_2", "text": "-pi/2 < alpha < pi/2", "parent_fact_ids": [], "is_final_answer": false, "origin": "given"}},
    {{"fact_id": "f_3", "text": "cos alpha = 4/5", "parent_fact_ids": ["f_1", "f_2"], "is_final_answer": false, "origin": "derived"}},
    {{"fact_id": "f_4", "text": "cot(2 alpha) = 7/24", "parent_fact_ids": ["f_1", "f_3"], "is_final_answer": true, "origin": "answer"}}
  ]
}}
""".strip()


def test_evaluator_scores_valid_graph_with_prover_signal() -> None:
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
                model_output=_valid_graph_output(),
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


def test_evaluator_uses_candidate_fdg_prompt_for_origin_schema() -> None:
    evaluator = FDGRLEvaluator(
        _config(),
        formalizer_bridge=FakeFormalizerBridge(),
        prover_bridge=FakeProverBridge(),
        lean_server=object(),
        owned_lean_server=False,
    )
    invalid = evaluator.prepare_graph_inputs(
        [
            CandidateGraphInput(
                record_id="legacy_origin_under_origin4",
                problem_text="problem",
                solution_or_cot="solution",
                model_output=_valid_graph_output(),
                extra_info={"fdg_prompt": "fdg_origin4"},
            )
        ]
    )[0]
    assert invalid.breakdown.validator_passed is False
    assert any(error["type"] == "invalid_origin" for error in invalid.breakdown.errors)

    valid = evaluator.prepare_graph_inputs(
        [
            CandidateGraphInput(
                record_id="origin4",
                problem_text="problem",
                solution_or_cot="solution",
                model_output=_valid_origin4_graph_output(),
                extra_info={"fdg_prompt": "fdg_origin4"},
            )
        ]
    )[0]
    assert valid.breakdown.validator_passed is True
    evaluator.close()


def test_evaluator_uses_config_fdg_prompt_as_fallback() -> None:
    evaluator = FDGRLEvaluator(
        _config(fdg_prompt="fdg_origin4"),
        formalizer_bridge=FakeFormalizerBridge(),
        prover_bridge=FakeProverBridge(),
        lean_server=object(),
        owned_lean_server=False,
    )
    prepared = evaluator.prepare_graph_inputs(
        [
            CandidateGraphInput(
                record_id="fallback_origin4",
                problem_text="problem",
                solution_or_cot="solution",
                model_output=_valid_graph_output(),
            )
        ]
    )[0]
    assert prepared.breakdown.validator_passed is False
    assert any(error["type"] == "invalid_origin" for error in prepared.breakdown.errors)
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


def test_evaluator_batches_facts_across_graphs() -> None:
    class RecordingFormalizerBridge(FakeFormalizerBridge):
        def __init__(self):
            self.calls = []

        async def batch_formalize(self, tasks):
            self.calls.append([(task.sample_index, task.fact["fact_id"]) for task in tasks])
            return await super().batch_formalize(tasks)

    class RecordingProverBridge(FakeProverBridge):
        def __init__(self):
            self.calls = []

        async def batch_prove(self, tasks):
            self.calls.append([(task.sample_index, task.fact["fact_id"]) for task in tasks])
            return await super().batch_prove(tasks)

    formalizer = RecordingFormalizerBridge()
    prover = RecordingProverBridge()
    evaluator = FDGRLEvaluator(
        _config(),
        formalizer_bridge=formalizer,
        prover_bridge=prover,
        lean_server=object(),
        owned_lean_server=False,
    )

    results = evaluator.evaluate_batch_sync(
        [
            CandidateGraphInput(
                record_id="graph_0",
                problem_text="problem",
                solution_or_cot="cot",
                model_output=_valid_graph_output("graph_0"),
            ),
            CandidateGraphInput(
                record_id="graph_1",
                problem_text="problem",
                solution_or_cot="cot",
                model_output=_valid_graph_output("graph_1"),
            ),
        ]
    )

    assert len(results) == 2
    assert [len(call) for call in formalizer.calls] == [4]
    assert [len(call) for call in prover.calls] == [4]
    assert {sample_index for call in formalizer.calls for sample_index, _fact_id in call} == {0, 1}
    evaluator.close()


def test_evaluator_pipelines_prover_after_formalizer_chunk() -> None:
    events = []
    prover_started = asyncio.Event()

    class PipelinedFormalizerBridge(FakeFormalizerBridge):
        def __init__(self):
            self.calls = 0

        async def batch_formalize(self, tasks):
            self.calls += 1
            events.append(f"formalizer_{self.calls}_start")
            if self.calls == 2:
                events.append("formalizer_2_wait")
                await asyncio.wait_for(prover_started.wait(), timeout=1.0)
            events.append(f"formalizer_{self.calls}_end")
            return await super().batch_formalize(tasks)

    class PipelinedProverBridge(FakeProverBridge):
        async def batch_prove(self, tasks):
            events.append("prover_start")
            prover_started.set()
            await asyncio.sleep(0)
            events.append("prover_end")
            return await super().batch_prove(tasks)

    evaluator = FDGRLEvaluator(
        _config(formalizer_batch_size=1, prover_batch_size=8),
        formalizer_bridge=PipelinedFormalizerBridge(),
        prover_bridge=PipelinedProverBridge(),
        lean_server=object(),
        owned_lean_server=False,
    )

    result = evaluator.evaluate_batch_sync(
        [
            CandidateGraphInput(
                record_id="alpha_example",
                problem_text="problem",
                solution_or_cot="cot",
                model_output=_valid_graph_output(),
            )
        ]
    )[0]

    assert result.num_formalized == 2
    assert events.index("prover_start") < events.index("formalizer_2_end")
    evaluator.close()
