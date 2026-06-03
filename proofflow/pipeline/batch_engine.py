from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, List, Optional, TypeVar


QueueItemT = TypeVar("QueueItemT")
TaskT = TypeVar("TaskT")
GenerationT = TypeVar("GenerationT")


@dataclass
class BatchEngineMetrics:
    idle_wait_seconds: float = 0.0
    generated_batches: int = 0
    generated_items: int = 0
    validated_items: int = 0


PrepareBatch = Callable[[List[QueueItemT]], Awaitable[List[TaskT]]]
GenerateBatch = Callable[[int, List[TaskT]], Awaitable[List[GenerationT]]]
ApplyGeneration = Callable[[TaskT, GenerationT], Awaitable[None]]
DonePredicate = Callable[[], Awaitable[bool]]


class BatchEngine(Generic[QueueItemT, TaskT, GenerationT]):
    """Reusable async queue engine for node-level LLM stages."""

    def __init__(
        self,
        *,
        name: str,
        worker_count: int,
        batch_size: int,
        wait_ms: int,
        max_pending_validation_batches: int,
        prepare_batch: PrepareBatch[QueueItemT, TaskT],
        generate_batch: GenerateBatch[TaskT, GenerationT],
        apply_generation: ApplyGeneration[TaskT, GenerationT],
        done: DonePredicate,
    ) -> None:
        self.name = name
        self.worker_count = max(1, int(worker_count))
        self.batch_size = max(1, int(batch_size))
        self.wait_ms = max(1, int(wait_ms))
        self.prepare_batch = prepare_batch
        self.generate_batch = generate_batch
        self.apply_generation = apply_generation
        self.done = done
        self.ready_queue: asyncio.Queue[QueueItemT] = asyncio.Queue()
        self.validation_queue: asyncio.Queue[tuple[TaskT, GenerationT]] = asyncio.Queue()
        self.validation_backpressure = asyncio.Semaphore(
            max(1, int(max_pending_validation_batches)) * self.batch_size
        )
        self.validation_workers: List[asyncio.Task[Any]] = []
        self.running_validation_items = 0
        self.validation_error: Optional[BaseException] = None
        self.metrics = BatchEngineMetrics()

    async def put(self, item: QueueItemT) -> None:
        await self.ready_queue.put(item)

    def _pending_validation_items(self) -> int:
        return self.validation_queue.qsize() + self.running_validation_items

    async def _pop_batch(self) -> List[QueueItemT]:
        timeout = self.wait_ms / 1000.0
        started = time.perf_counter()
        try:
            first = await asyncio.wait_for(self.ready_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.metrics.idle_wait_seconds += time.perf_counter() - started
            return []
        self.metrics.idle_wait_seconds += time.perf_counter() - started
        items = [first]
        while len(items) < self.batch_size:
            try:
                items.append(self.ready_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def _raise_validation_error(self) -> None:
        if self.validation_error is not None:
            raise RuntimeError(f"{self.name} validation worker failed") from self.validation_error

    async def _worker(self, worker_id: int) -> None:
        while True:
            self._raise_validation_error()
            batch_items = await self._pop_batch()
            if batch_items:
                tasks = await self.prepare_batch(batch_items)
                if tasks:
                    generations = await self.generate_batch(worker_id, tasks)
                    self.metrics.generated_batches += 1
                    self.metrics.generated_items += len(generations)
                    for task, generation in zip(tasks, generations):
                        await self.validation_backpressure.acquire()
                        await self.validation_queue.put((task, generation))
                continue
            if await self.done() and self._pending_validation_items() == 0:
                return
            await asyncio.sleep(self.wait_ms / 1000.0)

    async def _validation_worker(self) -> None:
        while True:
            task, generation = await self.validation_queue.get()
            self.running_validation_items += 1
            try:
                await self.apply_generation(task, generation)
                self.metrics.validated_items += 1
            except Exception as exc:
                self.validation_error = exc
            finally:
                self.running_validation_items -= 1
                self.validation_queue.task_done()
                self.validation_backpressure.release()

    async def run(self, *, validation_worker_count: int) -> None:
        self.validation_workers = [
            asyncio.create_task(self._validation_worker())
            for _ in range(max(1, int(validation_worker_count)))
        ]
        try:
            await asyncio.gather(*(self._worker(worker_id) for worker_id in range(self.worker_count)))
        finally:
            for worker in self.validation_workers:
                worker.cancel()
            await asyncio.gather(*self.validation_workers, return_exceptions=True)
            self.validation_workers.clear()
