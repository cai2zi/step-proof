from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from proofflow.fdg_stage_common import build_fdg_prove_messages
from proofflow.lean_check import LeanServer
from proofflow.llm_worker import LLMWorkerClient, LLMWorkerConfig
from proofflow.runtime_common import extract_last_lean_block

from .reward_types import (
    BridgeFactResult,
    BridgeFactTask,
    BridgeGenerationResult,
    LeanRuntimeConfig,
    ModelRuntimeConfig,
)


def _capture_conversation_enabled() -> bool:
    return os.getenv("RL_FDG_COT_TRACE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


class ProverBridge:
    def __init__(
        self,
        config: ModelRuntimeConfig,
        *,
        lean_config: LeanRuntimeConfig,
        lean_server: Optional[LeanServer] = None,
        owned_lean_server: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.lean_config = lean_config
        self.client: Optional[LLMWorkerClient] = None
        self.clients: List[LLMWorkerClient] = []
        self.lean_server: Optional[LeanServer] = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.lean_semaphore = asyncio.Semaphore(max(1, lean_config.check_concurrency))

    def _worker_gpu_groups(self) -> List[str]:
        num_workers = max(1, int(getattr(self.config, "num_workers", 1) or 1))
        tp = max(1, int(self.config.tensor_parallel_size))
        if num_workers == 1:
            return [self.config.gpus]

        gpu_ids = [gpu.strip() for gpu in str(self.config.gpus).split(",") if gpu.strip()]
        required = num_workers * tp
        if len(gpu_ids) < required:
            raise ValueError(
                "Not enough prover GPUs for parallel workers: "
                f"gpus={self.config.gpus!r}, tensor_parallel_size={tp}, "
                f"num_workers={num_workers}, required_gpus={required}"
            )
        return [
            ",".join(gpu_ids[index * tp : (index + 1) * tp])
            for index in range(num_workers)
        ]

    async def start(self) -> None:
        if not self.clients:
            gpu_groups = self._worker_gpu_groups()
            worker_configs = [
                LLMWorkerConfig(
                    name="rl_prover" if len(gpu_groups) == 1 else f"rl_prover_{index}",
                    gpus=gpus,
                    model_path=self.config.model_path,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    token_limit=self.config.token_limit,
                    dtype="float16",
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    top_p=self.config.top_p,
                    presence_penalty=self.config.presence_penalty,
                    frequency_penalty=self.config.frequency_penalty,
                    seed=self.config.seed,
                    top_k=self.config.top_k,
                    chat_template_kwargs=dict(self.config.chat_template_kwargs),
                )
                for index, gpus in enumerate(gpu_groups)
            ]
            clients: List[LLMWorkerClient] = []
            startup_errors: List[str] = []
            startup_stagger_s = float(os.getenv("RL_PROVER_WORKER_START_STAGGER_S", "3"))
            for worker_config in worker_configs:
                try:
                    client = await asyncio.to_thread(LLMWorkerClient, config=worker_config)
                except Exception as exc:
                    startup_errors.append(
                        f"{worker_config.name} gpus={worker_config.gpus}: {exc}"
                    )
                    print(
                        f"[rl_prover] worker startup failed "
                        f"name={worker_config.name} gpus={worker_config.gpus} error={exc}",
                        flush=True,
                    )
                    continue
                clients.append(client)
                print(
                    f"[rl_prover] worker startup ready "
                    f"name={worker_config.name} gpus={worker_config.gpus}",
                    flush=True,
                )
                if startup_stagger_s > 0 and len(clients) < len(worker_configs):
                    await asyncio.sleep(startup_stagger_s)
            if startup_errors:
                if not clients:
                    raise RuntimeError(
                        "Failed to start any prover workers: " + "; ".join(startup_errors)
                    )
                print(
                    "Failed to start some prover workers; continuing with "
                    f"{len(clients)}/{len(worker_configs)} workers: "
                    + "; ".join(startup_errors),
                    flush=True,
                )
            self.clients.extend(clients)
            if self.clients:
                self.client = self.clients[0]
        if self.lean_server is None:
            pool_size = self.lean_config.worker_pool_size or self.lean_config.check_concurrency
            self.lean_server = LeanServer(
                project_path=self.lean_config.mathlib_path,
                backend=self.lean_config.backend,
                pool_size=pool_size,
                temp_root=self.lean_config.temp_dir,
            )

    async def aclose(self) -> None:
        for client in self.clients:
            client.close()
        self.clients = []
        self.client = None
        if self.owned_lean_server and self.lean_server is not None:
            await self.lean_server.aclose()
            self.lean_server = None

    async def _validate_one(self, lean_code: str, *, job_id: str) -> tuple[bool, bool, Any]:
        assert self.lean_server is not None
        async with self.lean_semaphore:
            return await self.lean_server.check_lean_string_async(
                lean_code,
                temp_root=self.lean_config.temp_dir,
                job_id=job_id,
            )

    @staticmethod
    def _retry_feedback_message(result: BridgeFactResult) -> Dict[str, str]:
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

    @staticmethod
    def _retry_task(task: BridgeFactTask, result: BridgeFactResult) -> BridgeFactTask:
        return BridgeFactTask(
            sample_index=task.sample_index,
            fact=dict(task.fact),
            attempt=result.attempts + 1,
            feedback_messages=list(task.feedback_messages)
            + [ProverBridge._retry_feedback_message(result)],
            attempt_history=list(result.attempt_history),
        )

    async def batch_generate_proofs_once(
        self,
        tasks: List[BridgeFactTask],
    ) -> List[BridgeGenerationResult]:
        if not tasks:
            return []
        await self.start()
        assert self.clients

        batch: List[Dict[str, Any]] = []
        for task in tasks:
            fact = dict(task.fact)
            batch.append(
                {
                    "sample_index": task.sample_index,
                    "fact_id": str(fact["fact_id"]),
                    "messages": build_fdg_prove_messages(fact, prompt_name=self.config.prompt_name)
                    + list(task.feedback_messages),
                    "history": list(task.attempt_history),
                    "attempt": int(task.attempt),
                }
            )

        outputs_by_index: List[Optional[str]] = [None] * len(batch)
        if len(self.clients) == 1:
            outputs = await self.clients[0].generate_async([item["messages"] for item in batch])
            outputs_by_index = list(outputs)
        else:
            worker_items: List[List[tuple[int, Dict[str, Any]]]] = [
                [] for _ in range(len(self.clients))
            ]
            for index, item in enumerate(batch):
                worker_items[index % len(self.clients)].append((index, item))

            async def _generate_on_worker(
                client: LLMWorkerClient,
                indexed_items: List[tuple[int, Dict[str, Any]]],
            ) -> tuple[List[int], List[Optional[str]]]:
                if not indexed_items:
                    return [], []
                indices = [index for index, _item in indexed_items]
                message_batches = [item["messages"] for _index, item in indexed_items]
                return indices, await client.generate_async(message_batches)

            worker_outputs = await asyncio.gather(
                *(
                    _generate_on_worker(client, indexed_items)
                    for client, indexed_items in zip(self.clients, worker_items)
                    if indexed_items
                )
            )
            for indices, outputs in worker_outputs:
                for index, output in zip(indices, outputs):
                    outputs_by_index[index] = output

        results: List[BridgeGenerationResult] = []
        for item, output in zip(batch, outputs_by_index):
            attempt_num = int(item["attempt"])
            history = list(item["history"])
            conversation = None
            if _capture_conversation_enabled():
                conversation = list(item["messages"]) + [{"role": "assistant", "content": output or ""}]
            if output is None:
                error_msg = "token_overflow"
                history.append({"attempt": attempt_num, "kind": "token_overflow", "error_msg": error_msg})
                results.append(
                    BridgeGenerationResult(
                        sample_index=item["sample_index"],
                        fact_id=item["fact_id"],
                        stage="prover",
                        attempts=attempt_num,
                        extracted=False,
                        lean_code="",
                        error_message=error_msg,
                        raw_output="",
                        attempt_history=history,
                        conversation=conversation,
                    )
                )
                continue

            try:
                lean_code = extract_last_lean_block(output)
            except Exception as exc:
                error_msg = str(exc)
                history.append({"attempt": attempt_num, "kind": "extract_error", "error_msg": error_msg})
                results.append(
                    BridgeGenerationResult(
                        sample_index=item["sample_index"],
                        fact_id=item["fact_id"],
                        stage="prover",
                        attempts=attempt_num,
                        extracted=False,
                        lean_code="",
                        error_message=error_msg,
                        raw_output=output,
                        attempt_history=history,
                        conversation=conversation,
                    )
                )
                continue

            results.append(
                BridgeGenerationResult(
                    sample_index=item["sample_index"],
                    fact_id=item["fact_id"],
                    stage="prover",
                    attempts=attempt_num,
                    extracted=True,
                    lean_code=lean_code,
                    error_message="",
                    raw_output=output,
                    attempt_history=history,
                    conversation=conversation,
                )
            )
        return results

    async def batch_prove(self, tasks: List[BridgeFactTask]) -> List[BridgeFactResult]:
        if not tasks:
            return []

        pending = list(tasks)
        finished: List[BridgeFactResult] = []
        while pending:
            chunk = pending[: self.config.batch_size]
            pending = pending[self.config.batch_size :]
            generations = await self.batch_generate_proofs_once(chunk)
            chunk_results: List[BridgeFactResult | None] = [None] * len(chunk)
            validation_jobs: List[tuple[int, BridgeGenerationResult]] = []
            for index, generation in enumerate(generations):
                if generation.extracted:
                    validation_jobs.append((index, generation))
                    continue
                chunk_results[index] = BridgeFactResult(
                    sample_index=generation.sample_index,
                    fact_id=generation.fact_id,
                    stage="prover",
                    attempts=generation.attempts,
                    success=False,
                    verified=False,
                    lean_code="",
                    error_message=generation.error_message,
                    raw_output=generation.raw_output,
                    attempt_history=list(generation.attempt_history),
                    conversation=generation.conversation,
                )

            if not validation_jobs:
                validation_results = []
            else:
                validation_results = await asyncio.gather(
                    *(
                        self._validate_one(
                            generation.lean_code,
                            job_id=(
                                f"rl_prove_{generation.sample_index}_"
                                f"{generation.fact_id}_{generation.attempts}"
                            ),
                        )
                        for _index, generation in validation_jobs
                    )
                )

            for (index, generation), (
                lean_pass,
                lean_verify,
                error_msg,
            ) in zip(validation_jobs, validation_results):
                history = list(generation.attempt_history)
                history.append(
                    {
                        "attempt": generation.attempts,
                        "kind": "lean_check",
                        "lean_pass": bool(lean_pass),
                        "lean_verify": bool(lean_verify),
                        "error_msg": error_msg,
                    }
                )
                chunk_results[index] = BridgeFactResult(
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
                    conversation=generation.conversation,
                )

            for task, result in zip(chunk, chunk_results):
                assert result is not None
                if result.verified or result.attempts >= self.config.retries:
                    finished.append(result)
                    continue
                pending.append(self._retry_task(task, result))

        return finished
