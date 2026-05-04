import asyncio
import os
from types import SimpleNamespace

from proofflow.rl_fdg import reward_fn
from proofflow.rl_fdg import runtime_actor
from proofflow.rl_fdg.evaluator import PreparedGraphInput
from proofflow.rl_fdg.reward_types import (
    BridgeFactTask,
    BridgeGenerationResult,
    CandidateGraphInput,
    FactRewardTrace,
    GraphRewardBreakdown,
    SchedulerRuntimeConfig,
    TraceRuntimeConfig,
)


def _fake_config(
    *,
    include_prover: bool = False,
    formalizer_retries: int = 1,
    prover_retries: int = 1,
    graph_wait_ms: int = 25,
    formalizer_wait_ms: int = 25,
    prover_wait_ms: int = 25,
    formalizer_fail_extract_once: bool = False,
    prover_fail_verify_once: bool = False,
    trace_enabled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler=SchedulerRuntimeConfig(
            graph_wait_ms=graph_wait_ms,
            max_graph_batch_size=8,
            formalizer_wait_ms=formalizer_wait_ms,
            prover_wait_ms=prover_wait_ms,
            max_pending_graphs=128,
        ),
        include_prover=include_prover,
        formalizer=SimpleNamespace(batch_size=8, retries=formalizer_retries),
        prover=SimpleNamespace(batch_size=8, retries=prover_retries),
        lean=SimpleNamespace(check_concurrency=1, temp_dir="dummy"),
        trace=TraceRuntimeConfig(enabled=trace_enabled, out_dir="dummy"),
        formalizer_fail_extract_once=formalizer_fail_extract_once,
        prover_fail_verify_once=prover_fail_verify_once,
    )


def _payload(record_id: str, *, model_output: str = "{}") -> dict:
    return {
        "record_id": record_id,
        "problem_text": "problem",
        "solution_or_cot": "solution",
        "model_output": model_output,
        "ground_truth": "",
        "data_source": "fdg",
        "extra_info": {},
    }


class FakeFormalizerBridge:
    def __init__(self, config):
        self.config = config
        self.calls = []

    async def batch_generate_formalizations_once(self, tasks):
        self.calls.append(
            [(task.sample_index, task.fact["fact_id"], task.attempt, len(task.feedback_messages)) for task in tasks]
        )
        results = []
        for task in tasks:
            fact_id = str(task.fact["fact_id"])
            if self.config.formalizer_fail_extract_once and task.attempt == 1:
                results.append(
                    BridgeGenerationResult(
                        sample_index=task.sample_index,
                        fact_id=fact_id,
                        stage="formalizer",
                        attempts=task.attempt,
                        extracted=False,
                        lean_code="",
                        error_message="extract failed",
                        raw_output="bad",
                        attempt_history=[{"attempt": task.attempt, "kind": "extract_error"}],
                    )
                )
                continue
            results.append(
                BridgeGenerationResult(
                    sample_index=task.sample_index,
                    fact_id=fact_id,
                    stage="formalizer",
                    attempts=task.attempt,
                    extracted=True,
                    lean_code=f"formalizer_code_{fact_id}_{task.attempt}",
                    error_message="",
                    raw_output="```lean\nexample : True := by trivial\n```",
                    attempt_history=list(task.attempt_history),
                )
            )
        await asyncio.sleep(0)
        return results


class FakeProverBridge:
    def __init__(self):
        self.calls = []

    async def batch_generate_proofs_once(self, tasks):
        self.calls.append(
            [(task.sample_index, task.fact["fact_id"], task.attempt, len(task.feedback_messages)) for task in tasks]
        )
        await asyncio.sleep(0)
        return [
            BridgeGenerationResult(
                sample_index=task.sample_index,
                fact_id=str(task.fact["fact_id"]),
                stage="prover",
                attempts=task.attempt,
                extracted=True,
                lean_code=f"prover_code_{task.fact['fact_id']}_{task.attempt}",
                error_message="",
                raw_output="```lean\nexample : True := by trivial\n```",
                attempt_history=list(task.attempt_history),
            )
            for task in tasks
        ]


class FakeLeanServer:
    def __init__(self, config):
        self.config = config

    async def check_lean_string_async(self, lean_code, *, temp_root, job_id):
        await asyncio.sleep(0)
        if self.config.prover_fail_verify_once and job_id.startswith("rl_prove_") and job_id.endswith("_1"):
            return True, False, "prove failed"
        return True, True, ""


class FakeRuntimeEvaluator:
    instances = []

    def __init__(self, config):
        self.config = config
        self.formalizer_bridge = FakeFormalizerBridge(config)
        self.prover_bridge = FakeProverBridge()
        self.lean_server = FakeLeanServer(config)
        FakeRuntimeEvaluator.instances.append(self)

    async def ensure_runtime(self):
        return None

    def prepare_graph_inputs(self, inputs, *, sample_indices=None):
        sample_indices = sample_indices or list(range(len(inputs)))
        prepared = []
        for input_index, (sample_index, item) in enumerate(zip(sample_indices, inputs)):
            assert isinstance(item, CandidateGraphInput)
            if item.model_output == "invalid":
                prepared.append(
                    PreparedGraphInput(
                        input_index=input_index,
                        sample_index=sample_index,
                        breakdown=GraphRewardBreakdown(
                            record_id=item.record_id,
                            score=-1.0,
                            structure_score=-1.0,
                            formalizer_score=0.0,
                            prover_score=0.0,
                            final_answer_score=0.0,
                            length_penalty=0.0,
                            valid_json=False,
                            validator_passed=False,
                            num_facts=0,
                            num_non_root_facts=0,
                            num_final_facts=0,
                            num_warnings=0,
                            num_formalized=0,
                            num_proved=0,
                            num_final_verified=0,
                            parse_error="invalid",
                        ),
                    )
                )
                continue

            fact_id = f"fact_{item.record_id}"
            trace = FactRewardTrace(
                fact_id=fact_id,
                text="derived fact",
                parent_fact_ids=["root"],
                is_final_answer=True,
                origin="derived",
                proof_obligation={"goal": "True"},
            )
            structure_score = {"a": 0.2, "b": 0.4}.get(item.record_id, 0.2)
            breakdown = GraphRewardBreakdown(
                record_id=item.record_id,
                score=0.0,
                structure_score=structure_score,
                formalizer_score=0.0,
                prover_score=0.0,
                final_answer_score=0.0,
                length_penalty=0.0,
                valid_json=True,
                validator_passed=True,
                num_facts=2,
                num_non_root_facts=0,
                num_final_facts=0,
                num_warnings=0,
                num_formalized=0,
                num_proved=0,
                num_final_verified=0,
                facts=[trace],
            )
            task = BridgeFactTask(
                sample_index=sample_index,
                fact={
                    "fact_id": fact_id,
                    "text": "derived fact",
                    "parent_fact_ids": ["root"],
                    "is_final_answer": True,
                    "origin": "derived",
                    "proof_obligation": {"goal": "True"},
                },
            )
            prepared.append(
                PreparedGraphInput(
                    input_index=input_index,
                    sample_index=sample_index,
                    breakdown=breakdown,
                    fact_lookup={(sample_index, fact_id): trace},
                    form_tasks=[task],
                )
            )
        return prepared

    def make_prove_task(self, result, fact_lookup):
        trace = fact_lookup[(result.sample_index, result.fact_id)]
        return BridgeFactTask(
            sample_index=result.sample_index,
            fact={
                "fact_id": trace.fact_id,
                "text": trace.text,
                "parent_fact_ids": list(trace.parent_fact_ids),
                "is_final_answer": trace.is_final_answer,
                "origin": trace.origin,
                "proof_obligation": dict(trace.proof_obligation),
                "formalization": {"lean_code": result.lean_code, "lean_pass": True},
            },
        )

    def finalize_breakdown(self, breakdown):
        facts = [fact for fact in breakdown.facts if fact.parent_fact_ids]
        breakdown.num_non_root_facts = len(facts)
        breakdown.num_formalized = sum(1 for fact in facts if fact.formalizer is not None and fact.formalizer.success)
        breakdown.num_proved = sum(1 for fact in facts if fact.prover is not None and fact.prover.verified)
        breakdown.num_final_verified = sum(
            1 for fact in facts if fact.is_final_answer and fact.prover is not None and fact.prover.verified
        )
        breakdown.score = breakdown.structure_score + breakdown.num_formalized + breakdown.num_proved
        return breakdown


def _install_fake_runtime(monkeypatch, config):
    FakeRuntimeEvaluator.instances = []
    monkeypatch.setenv("STEP_PROOF_RL_TRACE", "0")
    monkeypatch.setattr(runtime_actor, "load_evaluator_config", lambda _path: config)
    monkeypatch.setattr(runtime_actor, "FDGRLEvaluator", FakeRuntimeEvaluator)


def test_runtime_actor_batches_formalizer_across_concurrent_single_graph_requests(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, _fake_config(include_prover=False))

    async def run_requests():
        actor = runtime_actor.FDGRLRuntimeActor("dummy_reward_config.yaml")
        return await asyncio.gather(
            actor.evaluate([_payload("r0")]),
            actor.evaluate([_payload("r1")]),
        )

    first, second = asyncio.run(run_requests())
    evaluator = FakeRuntimeEvaluator.instances[-1]

    assert first[0]["score"] == 1.2
    assert second[0]["score"] == 1.2
    assert [len(call) for call in evaluator.formalizer_bridge.calls] == [2]


def test_runtime_actor_retries_formalizer_generation_with_high_priority(monkeypatch) -> None:
    _install_fake_runtime(
        monkeypatch,
        _fake_config(include_prover=False, formalizer_retries=2, formalizer_fail_extract_once=True),
    )

    async def run_request():
        actor = runtime_actor.FDGRLRuntimeActor("dummy_reward_config.yaml")
        return await actor.evaluate([_payload("retry_form")])

    result = asyncio.run(run_request())
    evaluator = FakeRuntimeEvaluator.instances[-1]

    assert result[0]["num_formalized"] == 1
    assert evaluator.formalizer_bridge.calls[0][0][2] == 1
    assert evaluator.formalizer_bridge.calls[1][0][2] == 2
    assert evaluator.formalizer_bridge.calls[1][0][3] == 1


def test_runtime_actor_retries_prover_and_returns_input_order(monkeypatch) -> None:
    _install_fake_runtime(
        monkeypatch,
        _fake_config(include_prover=True, prover_retries=2, prover_fail_verify_once=True),
    )

    async def run_request():
        actor = runtime_actor.FDGRLRuntimeActor("dummy_reward_config.yaml")
        return await actor.evaluate([_payload("a"), _payload("b")])

    result = asyncio.run(run_request())
    evaluator = FakeRuntimeEvaluator.instances[-1]

    assert [item["structure_score"] for item in result] == [0.2, 0.4]
    assert all(item["num_proved"] == 1 for item in result)
    assert [len(call) for call in evaluator.prover_bridge.calls] == [2, 2]
    assert all(task[2] == 2 for task in evaluator.prover_bridge.calls[1])


def test_runtime_actor_invalid_graph_skips_formalizer(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch, _fake_config(include_prover=True))

    async def run_request():
        actor = runtime_actor.FDGRLRuntimeActor("dummy_reward_config.yaml")
        return await actor.evaluate([_payload("bad", model_output="invalid")])

    result = asyncio.run(run_request())
    evaluator = FakeRuntimeEvaluator.instances[-1]

    assert result == [
        {
            "score": -1.0,
            "validator_passed": False,
            "valid_json": False,
            "structure_score": -1.0,
            "formalizer_score": 0.0,
            "prover_score": 0.0,
            "final_answer_score": 0.0,
            "length_penalty": 0.0,
            "num_facts": 0,
            "num_non_root_facts": 0,
            "num_formalized": 0,
            "num_proved": 0,
            "num_final_verified": 0,
            "num_warnings": 0,
            "parse_error": "invalid",
        }
    ]
    assert evaluator.formalizer_bridge.calls == []


def test_runtime_actor_enables_conversation_capture_from_trace_config(monkeypatch) -> None:
    monkeypatch.delenv("RL_FDG_COT_TRACE", raising=False)
    _install_fake_runtime(monkeypatch, _fake_config(include_prover=True, trace_enabled=True))

    runtime_actor.FDGRLRuntimeActor("dummy_reward_config.yaml")

    assert os.environ["RL_FDG_COT_TRACE"] == "1"


def test_compute_score_falls_back_when_actor_unavailable(monkeypatch) -> None:
    calls = []

    def raise_actor(*args, **kwargs):
        raise RuntimeError("actor unavailable")

    def fake_local(batch_inputs, *, reward_config_path, reward_kwargs):
        calls.append((batch_inputs, reward_config_path, reward_kwargs))
        return [{"score": 0.5, "record_id": batch_inputs[0].record_id}]

    monkeypatch.setenv("STEP_PROOF_RL_TRACE", "0")
    monkeypatch.setattr(reward_fn, "_compute_with_actor", raise_actor)
    monkeypatch.setattr(reward_fn, "_compute_local", fake_local)

    result = reward_fn.compute_score(
        data_sources=["fdg"],
        solution_strs=["{}"],
        ground_truths=[""],
        extra_infos=[{"record_id": "fallback_case"}],
        reward_config_path="dummy_reward_config.yaml",
        use_runtime_actor=True,
        runtime_actor_fallback_to_local=True,
    )

    assert result == [{"score": 0.5, "record_id": "fallback_case"}]
    assert calls[0][0][0].record_id == "fallback_case"
