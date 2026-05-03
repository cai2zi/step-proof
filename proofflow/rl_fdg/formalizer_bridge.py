from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from proofflow.fdg_stage_common import build_fdg_form_messages
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


class FormalizerBridge:
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
        self.lean_server: Optional[LeanServer] = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.lean_semaphore = asyncio.Semaphore(max(1, lean_config.check_concurrency))

    async def start(self) -> None:
        if self.client is None:
            self.client = LLMWorkerClient(
                config=LLMWorkerConfig(
                    name="rl_formalizer",
                    gpus=self.config.gpus,
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
            )
        if self.lean_server is None:
            pool_size = self.lean_config.worker_pool_size or self.lean_config.check_concurrency
            self.lean_server = LeanServer(
                project_path=self.lean_config.mathlib_path,
                backend=self.lean_config.backend,
                pool_size=pool_size,
                temp_root=self.lean_config.temp_dir,
            )

    async def aclose(self) -> None:
        if self.client is not None:
            self.client.close()
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
        return {"role": "user", "content": content}

    @staticmethod
    def _retry_task(task: BridgeFactTask, result: BridgeFactResult) -> BridgeFactTask:
        return BridgeFactTask(
            sample_index=task.sample_index,
            fact=dict(task.fact),
            attempt=result.attempts + 1,
            feedback_messages=list(task.feedback_messages)
            + [FormalizerBridge._retry_feedback_message(result)],
            attempt_history=list(result.attempt_history),
        )

    async def batch_generate_formalizations_once(
        self,
        tasks: List[BridgeFactTask],
    ) -> List[BridgeGenerationResult]:
        if not tasks:
            return []
        await self.start()
        assert self.client is not None

        batch: List[Dict[str, Any]] = []
        for task in tasks:
            fact = dict(task.fact)
            batch.append(
                {
                    "sample_index": task.sample_index,
                    "fact_id": str(fact["fact_id"]),
                    "messages": build_fdg_form_messages(fact, prompt_name=self.config.prompt_name)
                    + list(task.feedback_messages),
                    "history": list(task.attempt_history),
                    "attempt": int(task.attempt),
                }
            )

        outputs = await self.client.generate_async([item["messages"] for item in batch])
        results: List[BridgeGenerationResult] = []
        for item, output in zip(batch, outputs):
            attempt_num = int(item["attempt"])
            history = list(item["history"])
            if output is None:
                error_msg = "token_overflow"
                history.append({"attempt": attempt_num, "kind": "token_overflow", "error_msg": error_msg})
                results.append(
                    BridgeGenerationResult(
                        sample_index=item["sample_index"],
                        fact_id=item["fact_id"],
                        stage="formalizer",
                        attempts=attempt_num,
                        extracted=False,
                        lean_code="",
                        error_message=error_msg,
                        raw_output="",
                        attempt_history=history,
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
                        stage="formalizer",
                        attempts=attempt_num,
                        extracted=False,
                        lean_code="",
                        error_message=error_msg,
                        raw_output=output,
                        attempt_history=history,
                    )
                )
                continue

            results.append(
                BridgeGenerationResult(
                    sample_index=item["sample_index"],
                    fact_id=item["fact_id"],
                    stage="formalizer",
                    attempts=attempt_num,
                    extracted=True,
                    lean_code=lean_code,
                    error_message="",
                    raw_output=output,
                    attempt_history=history,
                )
            )
        return results

    async def batch_formalize(self, tasks: List[BridgeFactTask]) -> List[BridgeFactResult]:
        if not tasks:
            return []

        pending = list(tasks)
        finished: List[BridgeFactResult] = []
        while pending:
            chunk = pending[: self.config.batch_size]
            pending = pending[self.config.batch_size :]
            generations = await self.batch_generate_formalizations_once(chunk)
            chunk_results: List[BridgeFactResult | None] = [None] * len(chunk)
            validation_jobs: List[tuple[int, BridgeGenerationResult]] = []
            for index, generation in enumerate(generations):
                if generation.extracted:
                    validation_jobs.append((index, generation))
                    continue
                chunk_results[index] = BridgeFactResult(
                    sample_index=generation.sample_index,
                    fact_id=generation.fact_id,
                    stage="formalizer",
                    attempts=generation.attempts,
                    success=False,
                    verified=False,
                    lean_code="",
                    error_message=generation.error_message,
                    raw_output=generation.raw_output,
                    attempt_history=list(generation.attempt_history),
                )

            if not validation_jobs:
                validation_results = []
            else:
                validation_results = await asyncio.gather(
                    *(
                        self._validate_one(
                            generation.lean_code,
                            job_id=(
                                f"rl_form_{generation.sample_index}_"
                                f"{generation.fact_id}_{generation.attempts}"
                            ),
                        )
                        for _index, generation in validation_jobs
                    )
                )

            for (index, generation), (
                lean_pass,
                _lean_verify,
                error_msg,
            ) in zip(validation_jobs, validation_results):
                history = list(generation.attempt_history)
                history.append(
                    {
                        "attempt": generation.attempts,
                        "kind": "lean_check",
                        "lean_pass": bool(lean_pass),
                        "error_msg": error_msg,
                    }
                )
                chunk_results[index] = BridgeFactResult(
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

            for task, result in zip(chunk, chunk_results):
                assert result is not None
                if result.success or result.attempts >= self.config.retries:
                    finished.append(result)
                    continue
                pending.append(self._retry_task(task, result))

        return finished
