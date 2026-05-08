from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from proofflow.fdg_graph import build_proof_obligation_from_fact
from proofflow.lean_check import LeanServer

from .formalizer_bridge import FormalizerBridge
from .parser import parse_fdg_candidate
from .prover_bridge import ProverBridge
from .reward_components import (
    count_truthy,
    score_final_answers,
    score_formalizer_pass,
    score_prover_pass,
    score_structure,
)
from .reward_types import (
    BridgeFactTask,
    CandidateGraphInput,
    FDGRLEvaluatorConfig,
    FactRewardTrace,
    GraphRewardBreakdown,
    LeanRuntimeConfig,
    ModelRuntimeConfig,
    RewardWeights,
    SchedulerRuntimeConfig,
    TraceRuntimeConfig,
)


def _scheduler_trace_enabled() -> bool:
    return os.getenv("STEP_PROOF_RL_SCHED_TRACE", os.getenv("STEP_PROOF_RL_TRACE", "1")) != "0"


def _scheduler_trace(message: str) -> None:
    if _scheduler_trace_enabled():
        print(f"[rl_fdg_scheduler] {message}", flush=True)


def _coerce_model_runtime_config(raw: Dict[str, Any]) -> ModelRuntimeConfig:
    cfg = dict(raw)
    for key in (
        "tensor_parallel_size",
        "max_tokens",
        "token_limit",
        "seed",
        "top_k",
        "retries",
        "batch_size",
        "num_workers",
    ):
        if key in cfg:
            cfg[key] = int(cfg[key])
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "gpu_memory_utilization"):
        if key in cfg:
            cfg[key] = float(cfg[key])
    return ModelRuntimeConfig(**cfg)


def _coerce_lean_runtime_config(raw: Dict[str, Any]) -> LeanRuntimeConfig:
    cfg = dict(raw)
    for key in ("check_concurrency", "worker_pool_size"):
        if key in cfg:
            cfg[key] = int(cfg[key])
    return LeanRuntimeConfig(**cfg)


def _coerce_scheduler_runtime_config(raw: Dict[str, Any]) -> SchedulerRuntimeConfig:
    cfg = dict(raw)
    for key in (
        "graph_wait_ms",
        "max_graph_batch_size",
        "formalizer_wait_ms",
        "prover_wait_ms",
        "max_pending_graphs",
        "runtime_actor_max_concurrency",
    ):
        if key in cfg:
            cfg[key] = int(cfg[key])
    return SchedulerRuntimeConfig(**cfg)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "on"}:
            return True
        if normalized in {"false", "no", "n", "off", ""}:
            return False
        try:
            return float(normalized) > 0
        except ValueError:
            return False
    return bool(value)


def _coerce_trace_runtime_config(raw: Dict[str, Any]) -> TraceRuntimeConfig:
    cfg = dict(raw)
    if "enabled" in cfg:
        cfg["enabled"] = _coerce_bool(cfg["enabled"])
    return TraceRuntimeConfig(**cfg)


def load_evaluator_config(config_path: str | Path) -> FDGRLEvaluatorConfig:
    cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    weights = RewardWeights(**dict(cfg.get("weights") or {}))
    runtime = dict(cfg.get("runtime") or {})
    formalizer = _coerce_model_runtime_config(dict(runtime.get("formalizer") or {}))
    prover = _coerce_model_runtime_config(dict(runtime.get("prover") or {}))
    lean = _coerce_lean_runtime_config(dict(runtime.get("lean") or {}))
    scheduler = _coerce_scheduler_runtime_config(dict(runtime.get("scheduler") or {}))
    trace = _coerce_trace_runtime_config(dict(runtime.get("trace") or {}))
    return FDGRLEvaluatorConfig(
        weights=weights,
        formalizer=formalizer,
        prover=prover,
        lean=lean,
        scheduler=scheduler,
        trace=trace,
        include_prover=_coerce_bool(runtime.get("include_prover", True)),
        fdg_prompt=str(runtime.get("fdg_prompt") or "fdg"),
    )


@dataclass
class PreparedGraphInput:
    input_index: int
    sample_index: int
    breakdown: GraphRewardBreakdown
    fact_lookup: Dict[tuple[int, str], FactRewardTrace] = field(default_factory=dict)
    form_tasks: List[BridgeFactTask] = field(default_factory=list)


class FDGRLEvaluator:
    def __init__(
        self,
        config: FDGRLEvaluatorConfig,
        *,
        formalizer_bridge: Optional[FormalizerBridge] = None,
        prover_bridge: Optional[ProverBridge] = None,
        lean_server: Optional[LeanServer] = None,
        owned_lean_server: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.lean_server = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.formalizer_bridge = formalizer_bridge
        self.prover_bridge = prover_bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    async def _ensure_runtime(self) -> None:
        await self.ensure_formalizer_runtime()
        if self.config.include_prover:
            await self.ensure_prover_runtime()

    async def ensure_lean_runtime(self) -> None:
        need_lean_server = self.formalizer_bridge is None or (
            self.config.include_prover and self.prover_bridge is None
        )
        if need_lean_server and self.lean_server is None:
            pool_size = self.config.lean.worker_pool_size or self.config.lean.check_concurrency
            self.lean_server = LeanServer(
                project_path=self.config.lean.mathlib_path,
                backend=self.config.lean.backend,
                pool_size=pool_size,
                temp_root=self.config.lean.temp_dir,
            )

    async def ensure_formalizer_runtime(self) -> None:
        await self.ensure_lean_runtime()
        if self.formalizer_bridge is None:
            self.formalizer_bridge = FormalizerBridge(
                self.config.formalizer,
                lean_config=self.config.lean,
                lean_server=self.lean_server,
                owned_lean_server=False,
            )
        await self.formalizer_bridge.start()

    async def ensure_prover_runtime(self) -> None:
        await self.ensure_lean_runtime()
        if self.config.include_prover:
            if self.prover_bridge is None:
                self.prover_bridge = ProverBridge(
                    self.config.prover,
                    lean_config=self.config.lean,
                    lean_server=self.lean_server,
                    owned_lean_server=False,
                )
            await self.prover_bridge.start()

    async def ensure_runtime(self) -> None:
        await self._ensure_runtime()

    async def aclose(self) -> None:
        if self.formalizer_bridge is not None:
            await self.formalizer_bridge.aclose()
            self.formalizer_bridge = None
        if self.prover_bridge is not None:
            await self.prover_bridge.aclose()
            self.prover_bridge = None
        if self.owned_lean_server and self.lean_server is not None:
            await self.lean_server.aclose()
            self.lean_server = None

    def close(self) -> None:
        loop = self._get_loop()
        loop.run_until_complete(self.aclose())
        loop.close()
        self._loop = None

    def prepare_graph_inputs(
        self,
        inputs: List[CandidateGraphInput],
        *,
        sample_indices: Optional[List[int]] = None,
    ) -> List[PreparedGraphInput]:
        if sample_indices is None:
            sample_indices = list(range(len(inputs)))
        if len(sample_indices) != len(inputs):
            raise ValueError(
                f"sample_indices length mismatch: expected {len(inputs)}, got {len(sample_indices)}"
            )

        parsed_items = [
            parse_fdg_candidate(
                item.model_output,
                prompt_name=str((item.extra_info or {}).get("fdg_prompt") or self.config.fdg_prompt or "fdg"),
            )
            for item in inputs
        ]
        prepared_items: List[PreparedGraphInput] = []

        for input_index, (sample_index, candidate, parsed) in enumerate(
            zip(sample_indices, inputs, parsed_items)
        ):
            errors = list(parsed.report.get("errors") or [])
            warnings = list(parsed.report.get("warnings") or [])
            raw_facts = list((parsed.raw_payload or {}).get("facts") or [])
            num_facts = len(raw_facts)

            if not parsed.valid_json:
                structure_score, length_penalty = score_structure(
                    valid_json=False,
                    validator_passed=False,
                    warning_count=0,
                    num_facts=0,
                    errors=errors,
                    weights=self.config.weights,
                )
                prepared_items.append(
                    PreparedGraphInput(
                        input_index=input_index,
                        sample_index=sample_index,
                        breakdown=GraphRewardBreakdown(
                            record_id=candidate.record_id,
                            score=structure_score,
                            structure_score=structure_score,
                            formalizer_score=0.0,
                            prover_score=0.0,
                            final_answer_score=0.0,
                            length_penalty=length_penalty,
                            valid_json=False,
                            validator_passed=False,
                            num_facts=0,
                            num_non_root_facts=0,
                            num_final_facts=0,
                            num_warnings=0,
                            num_formalized=0,
                            num_proved=0,
                            num_final_verified=0,
                            errors=errors,
                            warnings=[],
                            facts=[],
                            parse_error=parsed.parse_error or "",
                        ),
                    )
                )
                continue

            structure_score, length_penalty = score_structure(
                valid_json=True,
                validator_passed=parsed.validator_passed,
                warning_count=len(warnings),
                num_facts=num_facts,
                errors=errors,
                weights=self.config.weights,
            )
            if not parsed.validator_passed or parsed.document is None:
                prepared_items.append(
                    PreparedGraphInput(
                        input_index=input_index,
                        sample_index=sample_index,
                        breakdown=GraphRewardBreakdown(
                            record_id=candidate.record_id,
                            score=structure_score - length_penalty,
                            structure_score=structure_score,
                            formalizer_score=0.0,
                            prover_score=0.0,
                            final_answer_score=0.0,
                            length_penalty=length_penalty,
                            valid_json=True,
                            validator_passed=False,
                            num_facts=num_facts,
                            num_non_root_facts=0,
                            num_final_facts=0,
                            num_warnings=len(warnings),
                            num_formalized=0,
                            num_proved=0,
                            num_final_verified=0,
                            errors=errors,
                            warnings=warnings,
                            facts=[],
                        ),
                    )
                )
                continue

            traces: List[FactRewardTrace] = []
            fact_lookup: Dict[tuple[int, str], FactRewardTrace] = {}
            form_tasks: List[BridgeFactTask] = []
            for fact in parsed.document.facts:
                trace = FactRewardTrace(
                    fact_id=fact.fact_id,
                    text=fact.text,
                    parent_fact_ids=list(fact.parent_fact_ids),
                    is_final_answer=bool(fact.is_final_answer),
                    origin=fact.origin,
                    proof_obligation=(
                        {}
                        if not fact.parent_fact_ids
                        else build_proof_obligation_from_fact(parsed.document, fact.fact_id)
                    ),
                )
                traces.append(trace)
                fact_lookup[(sample_index, fact.fact_id)] = trace
                if fact.parent_fact_ids:
                    fact_state = fact.model_dump()
                    fact_state["proof_obligation"] = trace.proof_obligation
                    form_tasks.append(BridgeFactTask(sample_index=sample_index, fact=fact_state))

            prepared_items.append(
                PreparedGraphInput(
                    input_index=input_index,
                    sample_index=sample_index,
                    breakdown=GraphRewardBreakdown(
                        record_id=candidate.record_id,
                        score=0.0,
                        structure_score=structure_score,
                        formalizer_score=0.0,
                        prover_score=0.0,
                        final_answer_score=0.0,
                        length_penalty=length_penalty,
                        valid_json=True,
                        validator_passed=True,
                        num_facts=len(parsed.document.facts),
                        num_non_root_facts=0,
                        num_final_facts=0,
                        num_warnings=len(warnings),
                        num_formalized=0,
                        num_proved=0,
                        num_final_verified=0,
                        errors=errors,
                        warnings=warnings,
                        facts=traces,
                    ),
                    fact_lookup=fact_lookup,
                    form_tasks=form_tasks,
                )
            )

        return prepared_items

    @staticmethod
    def make_prove_task(
        result: Any,
        fact_lookup: Dict[tuple[int, str], FactRewardTrace],
    ) -> BridgeFactTask:
        trace = fact_lookup[(result.sample_index, result.fact_id)]
        fact_state = {
            "fact_id": trace.fact_id,
            "text": trace.text,
            "parent_fact_ids": list(trace.parent_fact_ids),
            "is_final_answer": trace.is_final_answer,
            "origin": trace.origin,
            "proof_obligation": dict(trace.proof_obligation),
            "formalization": {"lean_code": result.lean_code, "lean_pass": True},
        }
        return BridgeFactTask(sample_index=result.sample_index, fact=fact_state)

    def finalize_breakdown(self, breakdown: GraphRewardBreakdown) -> GraphRewardBreakdown:
        if not breakdown.validator_passed:
            return breakdown

        non_root_facts = [fact for fact in breakdown.facts if fact.parent_fact_ids]
        final_facts = [fact for fact in breakdown.facts if fact.is_final_answer]
        num_non_root = len(non_root_facts)
        num_final = len(final_facts)
        num_formalized = count_truthy(
            fact.formalizer is not None and fact.formalizer.success for fact in non_root_facts
        )
        num_proved = count_truthy(
            fact.prover is not None and fact.prover.verified for fact in non_root_facts
        )
        num_final_verified = count_truthy(
            fact.prover is not None and fact.prover.verified for fact in final_facts
        )

        formalizer_score = score_formalizer_pass(
            num_formalized=num_formalized,
            num_non_root_facts=num_non_root,
            weights=self.config.weights,
        )
        prover_score = score_prover_pass(
            num_proved=num_proved,
            num_non_root_facts=num_non_root,
            weights=self.config.weights,
        )
        final_score = score_final_answers(
            num_final_verified=num_final_verified,
            num_final_facts=num_final,
            weights=self.config.weights,
        )
        breakdown.formalizer_score = formalizer_score
        breakdown.prover_score = prover_score
        breakdown.final_answer_score = final_score
        breakdown.num_non_root_facts = num_non_root
        breakdown.num_final_facts = num_final
        breakdown.num_formalized = num_formalized
        breakdown.num_proved = num_proved
        breakdown.num_final_verified = num_final_verified
        breakdown.score = (
            breakdown.structure_score
            + formalizer_score
            + prover_score
            + final_score
            - breakdown.length_penalty
        )
        return breakdown

    @staticmethod
    def _task_chunks(tasks: List[BridgeFactTask], batch_size: int) -> List[List[BridgeFactTask]]:
        chunk_size = max(1, int(batch_size))
        return [tasks[start : start + chunk_size] for start in range(0, len(tasks), chunk_size)]

    @staticmethod
    async def _queue_batch(
        task_queue: asyncio.Queue[Any],
        *,
        batch_size: int,
        wait_ms: int,
        sentinel: Any,
    ) -> tuple[List[Any], bool]:
        first = await task_queue.get()
        if first is sentinel:
            return [], True

        batch = [first]
        saw_sentinel = False
        batch_size = max(1, int(batch_size))
        wait_s = max(0, int(wait_ms)) / 1000.0
        deadline = asyncio.get_running_loop().time() + wait_s

        while len(batch) < batch_size:
            if wait_s <= 0:
                try:
                    item = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(task_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

            if item is sentinel:
                saw_sentinel = True
                break
            batch.append(item)

        return batch, saw_sentinel

    @staticmethod
    def _make_prove_task(
        result: Any,
        fact_lookup: Dict[tuple[int, str], FactRewardTrace],
    ) -> BridgeFactTask:
        return FDGRLEvaluator.make_prove_task(result, fact_lookup)

    async def _run_formalizer_only(
        self,
        form_tasks: List[BridgeFactTask],
        fact_lookup: Dict[tuple[int, str], FactRewardTrace],
    ) -> None:
        assert self.formalizer_bridge is not None
        for chunk in self._task_chunks(form_tasks, self.config.formalizer.batch_size):
            start = time.perf_counter()
            _scheduler_trace(
                "formalizer dispatch "
                f"mode=formalizer_only batch_size={len(chunk)} "
                f"configured_batch_size={self.config.formalizer.batch_size}"
            )
            form_results = await self.formalizer_bridge.batch_formalize(chunk)
            success_count = sum(1 for result in form_results if result.success)
            _scheduler_trace(
                "formalizer done "
                f"mode=formalizer_only batch_size={len(chunk)} success={success_count} "
                f"elapsed_s={time.perf_counter() - start:.2f}"
            )
            for result in form_results:
                fact_lookup[(result.sample_index, result.fact_id)].formalizer = result

    async def _run_formalizer_prover_pipeline(
        self,
        form_tasks: List[BridgeFactTask],
        fact_lookup: Dict[tuple[int, str], FactRewardTrace],
    ) -> None:
        assert self.formalizer_bridge is not None
        assert self.prover_bridge is not None

        form_queue: asyncio.Queue[Any] = asyncio.Queue()
        prove_queue: asyncio.Queue[Any] = asyncio.Queue()
        form_sentinel = object()
        prove_sentinel = object()

        async def formalizer_worker() -> None:
            while True:
                batch, done = await self._queue_batch(
                    form_queue,
                    batch_size=self.config.formalizer.batch_size,
                    wait_ms=self.config.scheduler.formalizer_wait_ms,
                    sentinel=form_sentinel,
                )
                if batch:
                    start = time.perf_counter()
                    _scheduler_trace(
                        "formalizer dispatch "
                        f"batch_size={len(batch)} form_queue_remaining={form_queue.qsize()} "
                        f"prove_queue_size={prove_queue.qsize()} "
                        f"configured_batch_size={self.config.formalizer.batch_size} "
                        f"wait_ms={self.config.scheduler.formalizer_wait_ms}"
                    )
                    form_results = await self.formalizer_bridge.batch_formalize(batch)
                    success_count = 0
                    for result in form_results:
                        fact_lookup[(result.sample_index, result.fact_id)].formalizer = result
                        if result.success:
                            success_count += 1
                            await prove_queue.put(self._make_prove_task(result, fact_lookup))
                    _scheduler_trace(
                        "formalizer done "
                        f"batch_size={len(batch)} success={success_count} "
                        f"prove_enqueued={success_count} prove_queue_size={prove_queue.qsize()} "
                        f"elapsed_s={time.perf_counter() - start:.2f}"
                    )
                if done:
                    _scheduler_trace("formalizer drained; sending prover sentinel")
                    await prove_queue.put(prove_sentinel)
                    break

        async def prover_worker() -> None:
            while True:
                batch, done = await self._queue_batch(
                    prove_queue,
                    batch_size=self.config.prover.batch_size,
                    wait_ms=self.config.scheduler.prover_wait_ms,
                    sentinel=prove_sentinel,
                )
                if batch:
                    start = time.perf_counter()
                    _scheduler_trace(
                        "prover dispatch "
                        f"batch_size={len(batch)} prove_queue_remaining={prove_queue.qsize()} "
                        f"configured_batch_size={self.config.prover.batch_size} "
                        f"wait_ms={self.config.scheduler.prover_wait_ms}"
                    )
                    prove_results = await self.prover_bridge.batch_prove(batch)
                    verified_count = sum(1 for result in prove_results if result.verified)
                    _scheduler_trace(
                        "prover done "
                        f"batch_size={len(batch)} verified={verified_count} "
                        f"elapsed_s={time.perf_counter() - start:.2f}"
                    )
                    for result in prove_results:
                        fact_lookup[(result.sample_index, result.fact_id)].prover = result
                if done:
                    _scheduler_trace("prover drained")
                    break

        formalizer_task = asyncio.create_task(formalizer_worker())
        prover_task = asyncio.create_task(prover_worker())
        try:
            for task in form_tasks:
                await form_queue.put(task)
            await form_queue.put(form_sentinel)
            await asyncio.gather(formalizer_task, prover_task)
        except Exception:
            formalizer_task.cancel()
            prover_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await formalizer_task
            with contextlib.suppress(asyncio.CancelledError):
                await prover_task
            raise

    async def evaluate_batch(self, inputs: List[CandidateGraphInput]) -> List[GraphRewardBreakdown]:
        if not inputs:
            return []

        prepared_items = self.prepare_graph_inputs(inputs)
        breakdowns = [prepared.breakdown for prepared in prepared_items]
        form_tasks: List[BridgeFactTask] = []
        fact_lookup: Dict[tuple[int, str], FactRewardTrace] = {}
        valid_graphs = 0
        for prepared in prepared_items:
            if prepared.breakdown.validator_passed:
                valid_graphs += 1
            form_tasks.extend(prepared.form_tasks)
            fact_lookup.update(prepared.fact_lookup)

        if valid_graphs == 0:
            _scheduler_trace(
                f"evaluate_batch parsed inputs={len(inputs)} valid_graphs=0 form_tasks=0 "
                f"include_prover={self.config.include_prover}"
            )
            return breakdowns

        _scheduler_trace(
            "evaluate_batch parsed "
            f"inputs={len(inputs)} valid_graphs={valid_graphs} form_tasks={len(form_tasks)} "
            f"include_prover={self.config.include_prover} "
            f"formalizer_batch_size={self.config.formalizer.batch_size} "
            f"formalizer_wait_ms={self.config.scheduler.formalizer_wait_ms} "
            f"prover_batch_size={self.config.prover.batch_size} "
            f"prover_wait_ms={self.config.scheduler.prover_wait_ms}"
        )

        if form_tasks:
            await self.ensure_runtime()
            assert self.formalizer_bridge is not None
            if self.config.include_prover and self.prover_bridge is not None:
                await self._run_formalizer_prover_pipeline(form_tasks, fact_lookup)
            else:
                await self._run_formalizer_only(form_tasks, fact_lookup)

        for breakdown in breakdowns:
            self.finalize_breakdown(breakdown)

        return breakdowns

    def evaluate_batch_sync(self, inputs: List[CandidateGraphInput]) -> List[GraphRewardBreakdown]:
        loop = self._get_loop()
        return loop.run_until_complete(self.evaluate_batch(inputs))
