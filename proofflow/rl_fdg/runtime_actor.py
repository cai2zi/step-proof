from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from proofflow.rl_fdg.evaluator import FDGRLEvaluator, load_evaluator_config
from proofflow.rl_fdg.reward_types import (
    BridgeFactResult,
    BridgeFactTask,
    BridgeGenerationResult,
    CandidateGraphInput,
    FactRewardTrace,
    GraphRewardBreakdown,
)


NEW_TASK_PRIORITY = 10
RETRY_TASK_PRIORITY = 0


def _scheduler_trace_enabled() -> bool:
    return os.getenv("STEP_PROOF_RL_SCHED_TRACE", os.getenv("STEP_PROOF_RL_TRACE", "1")) != "0"


def _scheduler_trace(message: str) -> None:
    if _scheduler_trace_enabled():
        print(f"[rl_fdg_runtime] {message}", flush=True)


@dataclass
class _RuntimeRequestState:
    request_id: int
    future: asyncio.Future
    graph_ids: List[int]
    results: List[Optional[dict]]
    start_time: float


@dataclass
class _QueuedGraphInput:
    request_id: int
    request_offset: int
    graph_id: int
    input_payload: CandidateGraphInput


@dataclass
class _RuntimeGraphState:
    graph_id: int
    request_id: int
    request_offset: int
    breakdown: GraphRewardBreakdown
    fact_lookup: Dict[tuple[int, str], FactRewardTrace]
    pending_facts: int
    start_time: float


@dataclass(order=True)
class _QueuedFactTask:
    priority: int
    seq: int
    graph_id: int = field(compare=False)
    fact_id: str = field(compare=False)
    task: BridgeFactTask = field(compare=False)


@dataclass
class _QueuedLeanCheck:
    graph_id: int
    queued_task: _QueuedFactTask
    generation: BridgeGenerationResult


def _actor_name(reward_config_path: str, base_name: str) -> str:
    config = load_evaluator_config(reward_config_path)
    config_payload = asdict(config) if is_dataclass(config) else repr(config)
    payload = {
        "config": config_payload,
        "path": str(Path(reward_config_path).resolve()),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    safe_base = re.sub(r"[^0-9A-Za-z_]+", "_", base_name).strip("_") or "fdg_rl_runtime"
    return f"{safe_base}_{digest}"


class FDGRLRuntimeActor:
    def __init__(self, reward_config_path: str) -> None:
        self.reward_config_path = str(reward_config_path)
        self.config = load_evaluator_config(self.reward_config_path)
        self.evaluator = FDGRLEvaluator(self.config)
        self.graph_wait_s = max(0, int(self.config.scheduler.graph_wait_ms)) / 1000.0
        self.max_graph_batch_size = max(1, int(self.config.scheduler.max_graph_batch_size))
        self.max_pending_graphs = max(1, int(self.config.scheduler.max_pending_graphs))

        self.graph_queue: asyncio.Queue[_QueuedGraphInput] | None = None
        self.form_queue: asyncio.PriorityQueue[_QueuedFactTask] | None = None
        self.lean_queue: asyncio.Queue[_QueuedLeanCheck] | None = None
        self.prove_queue: asyncio.PriorityQueue[_QueuedFactTask] | None = None
        self._lock: asyncio.Lock | None = None
        self._runtime_lock: asyncio.Lock | None = None

        self.requests: Dict[int, _RuntimeRequestState] = {}
        self.graphs: Dict[int, _RuntimeGraphState] = {}
        self._request_seq = 0
        self._graph_seq = 0
        self._task_seq = 0

        self._graph_worker_task: asyncio.Task | None = None
        self._formalizer_worker_task: asyncio.Task | None = None
        self._prover_worker_task: asyncio.Task | None = None
        self._lean_worker_tasks: List[asyncio.Task] = []

        _scheduler_trace(
            "actor init "
            f"reward_config_path={self.reward_config_path} "
            f"graph_wait_ms={self.config.scheduler.graph_wait_ms} "
            f"max_graph_batch_size={self.max_graph_batch_size} "
            f"max_pending_graphs={self.max_pending_graphs} "
            f"formalizer_wait_ms={self.config.scheduler.formalizer_wait_ms} "
            f"prover_wait_ms={self.config.scheduler.prover_wait_ms}"
        )

    def _ensure_async_primitives(self) -> None:
        if self.graph_queue is None:
            self.graph_queue = asyncio.Queue()
        if self.form_queue is None:
            self.form_queue = asyncio.PriorityQueue()
        if self.lean_queue is None:
            self.lean_queue = asyncio.Queue()
        if self.prove_queue is None:
            self.prove_queue = asyncio.PriorityQueue()
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._runtime_lock is None:
            self._runtime_lock = asyncio.Lock()

    async def _ensure_runtime(self) -> None:
        self._ensure_async_primitives()
        assert self._runtime_lock is not None
        async with self._runtime_lock:
            await self.evaluator.ensure_runtime()

    async def _ensure_lean_runtime(self) -> None:
        self._ensure_async_primitives()
        assert self._runtime_lock is not None
        if self.evaluator.lean_server is not None:
            return
        async with self._runtime_lock:
            if self.evaluator.lean_server is None:
                ensure_lean = getattr(self.evaluator, "ensure_lean_runtime", None)
                if ensure_lean is not None:
                    await ensure_lean()
                else:
                    await self.evaluator.ensure_runtime()

    async def _ensure_formalizer_runtime(self) -> None:
        self._ensure_async_primitives()
        assert self._runtime_lock is not None
        if (
            self.evaluator.formalizer_bridge is not None
            and getattr(self.evaluator.formalizer_bridge, "client", None) is not None
        ):
            return
        async with self._runtime_lock:
            ensure_formalizer = getattr(self.evaluator, "ensure_formalizer_runtime", None)
            if ensure_formalizer is not None:
                await ensure_formalizer()
            else:
                await self.evaluator.ensure_runtime()

    async def _ensure_prover_runtime(self) -> None:
        self._ensure_async_primitives()
        assert self._runtime_lock is not None
        if (
            self.evaluator.prover_bridge is not None
            and getattr(self.evaluator.prover_bridge, "client", None) is not None
        ):
            return
        async with self._runtime_lock:
            ensure_prover = getattr(self.evaluator, "ensure_prover_runtime", None)
            if ensure_prover is not None:
                await ensure_prover()
            else:
                await self.evaluator.ensure_runtime()

    def _ensure_workers(self) -> None:
        self._ensure_async_primitives()
        if self._graph_worker_task is None or self._graph_worker_task.done():
            self._graph_worker_task = asyncio.create_task(
                self._worker_guard("graph_worker", self._graph_worker())
            )
        if self._formalizer_worker_task is None or self._formalizer_worker_task.done():
            self._formalizer_worker_task = asyncio.create_task(
                self._worker_guard("formalizer_worker", self._formalizer_worker())
            )
        if self.config.include_prover and (
            self._prover_worker_task is None or self._prover_worker_task.done()
        ):
            self._prover_worker_task = asyncio.create_task(
                self._worker_guard("prover_worker", self._prover_worker())
            )

        expected_lean_workers = max(1, int(self.config.lean.check_concurrency))
        live_lean_workers = [task for task in self._lean_worker_tasks if not task.done()]
        self._lean_worker_tasks = live_lean_workers
        for index in range(len(live_lean_workers), expected_lean_workers):
            self._lean_worker_tasks.append(
                asyncio.create_task(
                    self._worker_guard(f"lean_worker_{index}", self._lean_worker(index))
                )
            )

    async def evaluate(self, input_payloads: List[dict]) -> List[dict]:
        if not input_payloads:
            return []

        self._ensure_async_primitives()
        self._ensure_workers()
        assert self._lock is not None
        assert self.graph_queue is not None

        inputs = [CandidateGraphInput(**payload) for payload in input_payloads]
        future = asyncio.get_running_loop().create_future()
        async with self._lock:
            active_graphs = len(self.graphs) + self.graph_queue.qsize()
            if active_graphs + len(inputs) > self.max_pending_graphs:
                raise RuntimeError(
                    "FDG runtime actor graph backlog exceeded: "
                    f"active={active_graphs}, incoming={len(inputs)}, max={self.max_pending_graphs}"
                )

            request_id = self._request_seq
            self._request_seq += 1
            graph_ids: List[int] = []
            queued_graphs: List[_QueuedGraphInput] = []
            for offset, candidate in enumerate(inputs):
                graph_id = self._graph_seq
                self._graph_seq += 1
                graph_ids.append(graph_id)
                queued_graphs.append(
                    _QueuedGraphInput(
                        request_id=request_id,
                        request_offset=offset,
                        graph_id=graph_id,
                        input_payload=candidate,
                    )
                )
            self.requests[request_id] = _RuntimeRequestState(
                request_id=request_id,
                future=future,
                graph_ids=graph_ids,
                results=[None] * len(inputs),
                start_time=time.perf_counter(),
            )
            for item in queued_graphs:
                self.graph_queue.put_nowait(item)
            pending_requests = len(self.requests)
            pending_graphs = len(self.graphs) + self.graph_queue.qsize()

        _scheduler_trace(
            "request enqueue "
            f"request_id={request_id} graphs={len(inputs)} "
            f"pending_requests={pending_requests} pending_graphs={pending_graphs}"
        )
        return await future

    async def _worker_guard(self, name: str, worker_coro) -> None:
        try:
            await worker_coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _scheduler_trace(f"{name} failed error={type(exc).__name__}: {exc}")
            await self._fail_all(exc)
            raise

    async def _fail_all(self, exc: Exception) -> None:
        self._ensure_async_primitives()
        assert self._lock is not None
        async with self._lock:
            requests = list(self.requests.values())
            self.requests.clear()
            self.graphs.clear()
        for request in requests:
            if not request.future.done():
                request.future.set_exception(exc)

    @staticmethod
    async def _queue_batch(
        queue: asyncio.Queue[Any],
        *,
        batch_size: int,
        wait_ms: int,
    ) -> List[Any]:
        first = await queue.get()
        batch = [first]
        batch_size = max(1, int(batch_size))
        wait_s = max(0, int(wait_ms)) / 1000.0
        deadline = asyncio.get_running_loop().time() + wait_s

        while len(batch) < batch_size:
            if wait_s <= 0:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            batch.append(item)
        return batch

    async def _graph_worker(self) -> None:
        assert self.graph_queue is not None
        while True:
            graph_batch = await self._queue_batch(
                self.graph_queue,
                batch_size=self.max_graph_batch_size,
                wait_ms=self.config.scheduler.graph_wait_ms,
            )
            inputs = [item.input_payload for item in graph_batch]
            sample_indices = [item.graph_id for item in graph_batch]
            prepared_items = self.evaluator.prepare_graph_inputs(
                inputs,
                sample_indices=sample_indices,
            )
            valid_count = sum(1 for item in prepared_items if item.breakdown.validator_passed)
            form_task_count = sum(len(item.form_tasks) for item in prepared_items)
            _scheduler_trace(
                "graph parsed "
                f"graphs={len(graph_batch)} valid={valid_count} form_tasks={form_task_count} "
                f"graph_queue={self.graph_queue.qsize()}"
            )

            for queued_graph, prepared in zip(graph_batch, prepared_items):
                if not prepared.breakdown.validator_passed or not prepared.form_tasks:
                    self.evaluator.finalize_breakdown(prepared.breakdown)
                    self._complete_graph(
                        queued_graph.graph_id,
                        queued_graph.request_id,
                        queued_graph.request_offset,
                        prepared.breakdown,
                        start_time=time.perf_counter(),
                    )
                    continue

                self.graphs[queued_graph.graph_id] = _RuntimeGraphState(
                    graph_id=queued_graph.graph_id,
                    request_id=queued_graph.request_id,
                    request_offset=queued_graph.request_offset,
                    breakdown=prepared.breakdown,
                    fact_lookup=prepared.fact_lookup,
                    pending_facts=len(prepared.form_tasks),
                    start_time=time.perf_counter(),
                )
                for task in prepared.form_tasks:
                    self._enqueue_form_task(queued_graph.graph_id, task, priority=NEW_TASK_PRIORITY)
                assert self.form_queue is not None
                _scheduler_trace(
                    "form enqueue "
                    f"graph_id={queued_graph.graph_id} tasks={len(prepared.form_tasks)} "
                    f"form_queue={self.form_queue.qsize()}"
                )

    async def _formalizer_worker(self) -> None:
        assert self.form_queue is not None
        assert self.lean_queue is not None

        while True:
            batch = await self._queue_batch(
                self.form_queue,
                batch_size=self.config.formalizer.batch_size,
                wait_ms=self.config.scheduler.formalizer_wait_ms,
            )
            await self._ensure_formalizer_runtime()
            assert self.evaluator.formalizer_bridge is not None
            retry_count = sum(1 for item in batch if item.task.attempt > 1)
            _scheduler_trace(
                "formalizer dispatch "
                f"batch_size={len(batch)} retry_tasks={retry_count} "
                f"form_queue={self.form_queue.qsize()}"
            )
            generations = await self.evaluator.formalizer_bridge.batch_generate_formalizations_once(
                [item.task for item in batch]
            )
            extracted_count = 0
            retry_enqueued = 0
            lean_enqueued = 0
            for queued_task, generation in zip(batch, generations):
                if generation.extracted:
                    extracted_count += 1
                    self.lean_queue.put_nowait(
                        _QueuedLeanCheck(
                            graph_id=queued_task.graph_id,
                            queued_task=queued_task,
                            generation=generation,
                        )
                    )
                    lean_enqueued += 1
                    continue
                result = self._fact_result_from_generation(generation)
                retry_enqueued += self._retry_or_finish_formalizer(queued_task, result)
            _scheduler_trace(
                "formalizer generated "
                f"batch_size={len(batch)} extracted={extracted_count} "
                f"retry={retry_enqueued} lean_enqueued={lean_enqueued}"
            )

    async def _prover_worker(self) -> None:
        assert self.prove_queue is not None
        assert self.lean_queue is not None

        while True:
            batch = await self._queue_batch(
                self.prove_queue,
                batch_size=self.config.prover.batch_size,
                wait_ms=self.config.scheduler.prover_wait_ms,
            )
            await self._ensure_prover_runtime()
            assert self.evaluator.prover_bridge is not None
            retry_count = sum(1 for item in batch if item.task.attempt > 1)
            _scheduler_trace(
                "prover dispatch "
                f"batch_size={len(batch)} retry_tasks={retry_count} "
                f"prove_queue={self.prove_queue.qsize()}"
            )
            generations = await self.evaluator.prover_bridge.batch_generate_proofs_once(
                [item.task for item in batch]
            )
            extracted_count = 0
            retry_enqueued = 0
            lean_enqueued = 0
            for queued_task, generation in zip(batch, generations):
                if generation.extracted:
                    extracted_count += 1
                    self.lean_queue.put_nowait(
                        _QueuedLeanCheck(
                            graph_id=queued_task.graph_id,
                            queued_task=queued_task,
                            generation=generation,
                        )
                    )
                    lean_enqueued += 1
                    continue
                result = self._fact_result_from_generation(generation)
                retry_enqueued += self._retry_or_finish_prover(queued_task, result)
            _scheduler_trace(
                "prover generated "
                f"batch_size={len(batch)} extracted={extracted_count} "
                f"retry={retry_enqueued} lean_enqueued={lean_enqueued}"
            )

    async def _lean_worker(self, worker_index: int) -> None:
        assert self.lean_queue is not None

        while True:
            item = await self.lean_queue.get()
            await self._ensure_lean_runtime()
            assert self.evaluator.lean_server is not None
            generation = item.generation
            job_prefix = "rl_form" if generation.stage == "formalizer" else "rl_prove"
            lean_pass, lean_verify, error_msg = await self.evaluator.lean_server.check_lean_string_async(
                generation.lean_code,
                temp_root=self.config.lean.temp_dir,
                job_id=f"{job_prefix}_{generation.sample_index}_{generation.fact_id}_{generation.attempts}",
            )
            history = list(generation.attempt_history)
            history.append(
                {
                    "attempt": generation.attempts,
                    "kind": "lean_check",
                    "lean_pass": bool(lean_pass),
                    "lean_verify": bool(lean_verify),
                    "error_msg": error_msg,
                    "worker_index": worker_index,
                }
            )
            if generation.stage == "formalizer":
                result = BridgeFactResult(
                    sample_index=generation.sample_index,
                    fact_id=generation.fact_id,
                    stage="formalizer",
                    attempts=generation.attempts,
                    success=bool(lean_pass),
                    verified=False,
                    lean_code=generation.lean_code,
                    error_message="" if lean_pass else str(error_msg),
                    raw_output=generation.raw_output,
                    attempt_history=history,
                )
                routed_to = self._handle_formalizer_lean_result(item.queued_task, result)
                _scheduler_trace(
                    "lean checked "
                    f"stage=formalizer pass={bool(lean_pass)} retry={routed_to == 'form_retry'} "
                    f"routed_to={routed_to}"
                )
            else:
                result = BridgeFactResult(
                    sample_index=generation.sample_index,
                    fact_id=generation.fact_id,
                    stage="prover",
                    attempts=generation.attempts,
                    success=bool(lean_pass),
                    verified=bool(lean_verify),
                    lean_code=generation.lean_code,
                    error_message="" if lean_verify else str(error_msg),
                    raw_output=generation.raw_output,
                    attempt_history=history,
                )
                routed_to = self._handle_prover_lean_result(item.queued_task, result)
                _scheduler_trace(
                    "lean checked "
                    f"stage=prover pass={bool(lean_verify)} retry={routed_to == 'prove_retry'} "
                    f"routed_to={routed_to}"
                )

    def _next_task_seq(self) -> int:
        seq = self._task_seq
        self._task_seq += 1
        return seq

    def _enqueue_form_task(self, graph_id: int, task: BridgeFactTask, *, priority: int) -> None:
        assert self.form_queue is not None
        self.form_queue.put_nowait(
            _QueuedFactTask(
                priority=priority,
                seq=self._next_task_seq(),
                graph_id=graph_id,
                fact_id=str(task.fact["fact_id"]),
                task=task,
            )
        )

    def _enqueue_prove_task(self, graph_id: int, task: BridgeFactTask, *, priority: int) -> None:
        assert self.prove_queue is not None
        self.prove_queue.put_nowait(
            _QueuedFactTask(
                priority=priority,
                seq=self._next_task_seq(),
                graph_id=graph_id,
                fact_id=str(task.fact["fact_id"]),
                task=task,
            )
        )

    @staticmethod
    def _fact_result_from_generation(generation: BridgeGenerationResult) -> BridgeFactResult:
        return BridgeFactResult(
            sample_index=generation.sample_index,
            fact_id=generation.fact_id,
            stage=generation.stage,
            attempts=generation.attempts,
            success=False,
            verified=False,
            lean_code="",
            error_message=generation.error_message,
            raw_output=generation.raw_output,
            attempt_history=list(generation.attempt_history),
        )

    @staticmethod
    def _retry_feedback_message(stage: str, result: BridgeFactResult) -> dict:
        if stage == "formalizer":
            if result.error_message == "token_overflow":
                content = "The previous response hit a token limit. Regenerate the full Lean4 formalization."
            elif not result.lean_code:
                content = (
                    f"The previous response was invalid because: {result.error_message}\n"
                    "Please regenerate valid Lean4 code."
                )
            else:
                content = (
                    "The previous Lean4 code did not compile.\n"
                    f"Lean errors: {result.error_message}\n"
                    "Please regenerate corrected Lean4 code."
                )
        else:
            if result.error_message == "token_overflow":
                content = "The previous response hit a token limit. Regenerate the complete Lean4 proof."
            elif not result.lean_code:
                content = (
                    f"The previous response was invalid because: {result.error_message}\n"
                    "Please regenerate valid Lean4 proof code."
                )
            else:
                content = (
                    "The previous Lean4 proof did not verify.\n"
                    f"Lean feedback: {result.error_message}\n"
                    "Please regenerate a corrected proof."
                )
        return {"role": "user", "content": content}

    def _make_retry_task(
        self,
        *,
        stage: str,
        queued_task: _QueuedFactTask,
        result: BridgeFactResult,
    ) -> BridgeFactTask:
        return BridgeFactTask(
            sample_index=queued_task.task.sample_index,
            fact=dict(queued_task.task.fact),
            attempt=result.attempts + 1,
            feedback_messages=list(queued_task.task.feedback_messages)
            + [self._retry_feedback_message(stage, result)],
            attempt_history=list(result.attempt_history),
        )

    def _get_graph(self, graph_id: int) -> Optional[_RuntimeGraphState]:
        return self.graphs.get(graph_id)

    def _retry_or_finish_formalizer(
        self,
        queued_task: _QueuedFactTask,
        result: BridgeFactResult,
    ) -> int:
        graph = self._get_graph(queued_task.graph_id)
        if graph is None:
            return 0
        trace = graph.fact_lookup.get((result.sample_index, result.fact_id))
        if trace is not None:
            trace.formalizer = result
        if result.attempts < self.config.formalizer.retries:
            retry_task = self._make_retry_task(
                stage="formalizer",
                queued_task=queued_task,
                result=result,
            )
            self._enqueue_form_task(queued_task.graph_id, retry_task, priority=RETRY_TASK_PRIORITY)
            return 1
        self._mark_fact_done(queued_task.graph_id, result.fact_id)
        return 0

    def _retry_or_finish_prover(
        self,
        queued_task: _QueuedFactTask,
        result: BridgeFactResult,
    ) -> int:
        graph = self._get_graph(queued_task.graph_id)
        if graph is None:
            return 0
        trace = graph.fact_lookup.get((result.sample_index, result.fact_id))
        if trace is not None:
            trace.prover = result
        if result.verified:
            self._mark_fact_done(queued_task.graph_id, result.fact_id)
            return 0
        if result.attempts < self.config.prover.retries:
            retry_task = self._make_retry_task(
                stage="prover",
                queued_task=queued_task,
                result=result,
            )
            self._enqueue_prove_task(queued_task.graph_id, retry_task, priority=RETRY_TASK_PRIORITY)
            return 1
        self._mark_fact_done(queued_task.graph_id, result.fact_id)
        return 0

    def _handle_formalizer_lean_result(
        self,
        queued_task: _QueuedFactTask,
        result: BridgeFactResult,
    ) -> str:
        graph = self._get_graph(queued_task.graph_id)
        if graph is None:
            return "dropped"
        trace = graph.fact_lookup.get((result.sample_index, result.fact_id))
        if trace is not None:
            trace.formalizer = result
        if not result.success:
            retry_count = self._retry_or_finish_formalizer(queued_task, result)
            return "form_retry" if retry_count else "fact_done"
        if not self.config.include_prover:
            self._mark_fact_done(queued_task.graph_id, result.fact_id)
            return "fact_done"
        prove_task = self.evaluator.make_prove_task(result, graph.fact_lookup)
        self._enqueue_prove_task(queued_task.graph_id, prove_task, priority=NEW_TASK_PRIORITY)
        assert self.prove_queue is not None
        return "prove_queue"

    def _handle_prover_lean_result(
        self,
        queued_task: _QueuedFactTask,
        result: BridgeFactResult,
    ) -> str:
        retry_count = self._retry_or_finish_prover(queued_task, result)
        if result.verified:
            return "fact_done"
        return "prove_retry" if retry_count else "fact_done"

    def _mark_fact_done(self, graph_id: int, fact_id: str) -> None:
        graph = self.graphs.get(graph_id)
        if graph is None:
            return
        graph.pending_facts -= 1
        if graph.pending_facts > 0:
            return
        breakdown = self.evaluator.finalize_breakdown(graph.breakdown)
        self.graphs.pop(graph_id, None)
        self._complete_graph(
            graph_id,
            graph.request_id,
            graph.request_offset,
            breakdown,
            start_time=graph.start_time,
        )

    def _complete_graph(
        self,
        graph_id: int,
        request_id: int,
        request_offset: int,
        breakdown: GraphRewardBreakdown,
        *,
        start_time: float,
    ) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        request.results[request_offset] = breakdown.to_reward_dict()
        _scheduler_trace(
            "graph done "
            f"graph_id={graph_id} request_id={request_id} score={breakdown.score:.4f} "
            f"elapsed_s={time.perf_counter() - start_time:.2f}"
        )
        if not all(result is not None for result in request.results):
            return
        self.requests.pop(request_id, None)
        if not request.future.done():
            request.future.set_result([result for result in request.results if result is not None])
        _scheduler_trace(
            "request done "
            f"request_id={request_id} graphs={len(request.graph_ids)} "
            f"elapsed_s={time.perf_counter() - request.start_time:.2f}"
        )


def get_or_create_runtime_actor(
    *,
    reward_config_path: str,
    runtime_actor_name: str = "fdg_rl_runtime",
    runtime_actor_namespace: str = "step_proof_rl",
):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized; cannot use shared FDG runtime actor.")

    actor_name = _actor_name(reward_config_path, runtime_actor_name)
    try:
        return ray.get_actor(actor_name, namespace=runtime_actor_namespace)
    except ValueError:
        pass

    remote_cls = ray.remote(num_gpus=0, max_concurrency=128)(FDGRLRuntimeActor)
    node_id = ray.get_runtime_context().get_node_id()
    options = {
        "name": actor_name,
        "namespace": runtime_actor_namespace,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=True),
    }
    try:
        return remote_cls.options(**options).remote(str(reward_config_path))
    except ValueError:
        return ray.get_actor(actor_name, namespace=runtime_actor_namespace)
