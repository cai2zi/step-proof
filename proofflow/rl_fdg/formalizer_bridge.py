from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from proofflow.fdg_stage_common import build_fdg_form_messages
from proofflow.lean_check import LeanServer
from proofflow.llm_worker import LLMWorkerClient, LLMWorkerConfig
from proofflow.runtime_common import extract_last_lean_block

from .reward_types import BridgeFactResult, BridgeFactTask, LeanRuntimeConfig, ModelRuntimeConfig


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
                    gpu_memory_utilization=0.9,
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

    async def batch_formalize(self, tasks: List[BridgeFactTask]) -> List[BridgeFactResult]:
        if not tasks:
            return []
        await self.start()
        assert self.client is not None

        pending: List[Dict[str, Any]] = []
        for task in tasks:
            fact = dict(task.fact)
            pending.append(
                {
                    "sample_index": task.sample_index,
                    "fact_id": str(fact["fact_id"]),
                    "messages": build_fdg_form_messages(fact, prompt_name=self.config.prompt_name),
                    "history": [],
                    "attempt": 1,
                }
            )

        finished: List[BridgeFactResult] = []
        while pending:
            batch = pending[: self.config.batch_size]
            pending = pending[self.config.batch_size :]
            outputs = self.client.generate([item["messages"] for item in batch])
            for item, output in zip(batch, outputs):
                attempt_num = int(item["attempt"])
                history = list(item["history"])
                if output is None:
                    error_msg = "token_overflow"
                    history.append({"attempt": attempt_num, "kind": "token_overflow", "error_msg": error_msg})
                    if attempt_num < self.config.retries:
                        item["history"] = history
                        item["attempt"] = attempt_num + 1
                        item["messages"] = list(item["messages"]) + [
                            {
                                "role": "user",
                                "content": "The previous response hit a token limit. Regenerate the full Lean4 formalization.",
                            }
                        ]
                        pending.append(item)
                    else:
                        finished.append(
                            BridgeFactResult(
                                sample_index=item["sample_index"],
                                fact_id=item["fact_id"],
                                stage="formalizer",
                                attempts=attempt_num,
                                success=False,
                                verified=False,
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
                    if attempt_num < self.config.retries:
                        item["history"] = history
                        item["attempt"] = attempt_num + 1
                        item["messages"] = list(item["messages"]) + [
                            {
                                "role": "user",
                                "content": f"The previous response was invalid because: {error_msg}\nPlease regenerate valid Lean4 code.",
                            }
                        ]
                        pending.append(item)
                    else:
                        finished.append(
                            BridgeFactResult(
                                sample_index=item["sample_index"],
                                fact_id=item["fact_id"],
                                stage="formalizer",
                                attempts=attempt_num,
                                success=False,
                                verified=False,
                                lean_code="",
                                error_message=error_msg,
                                raw_output=output,
                                attempt_history=history,
                            )
                        )
                    continue

                lean_pass, _lean_verify, error_msg = await self._validate_one(
                    lean_code,
                    job_id=f"rl_form_{item['sample_index']}_{item['fact_id']}_{attempt_num}",
                )
                history.append(
                    {
                        "attempt": attempt_num,
                        "kind": "lean_check",
                        "lean_pass": bool(lean_pass),
                        "error_msg": error_msg,
                    }
                )
                if lean_pass:
                    finished.append(
                        BridgeFactResult(
                            sample_index=item["sample_index"],
                            fact_id=item["fact_id"],
                            stage="formalizer",
                            attempts=attempt_num,
                            success=True,
                            verified=False,
                            lean_code=lean_code,
                            error_message="",
                            raw_output=output,
                            attempt_history=history,
                        )
                    )
                    continue

                if attempt_num < self.config.retries:
                    item["history"] = history
                    item["attempt"] = attempt_num + 1
                    item["messages"] = list(item["messages"]) + [
                        {
                            "role": "user",
                            "content": (
                                "The previous Lean4 code did not compile.\n"
                                f"Lean errors: {error_msg}\n"
                                "Please regenerate corrected Lean4 code."
                            ),
                        }
                    ]
                    pending.append(item)
                    continue

                finished.append(
                    BridgeFactResult(
                        sample_index=item["sample_index"],
                        fact_id=item["fact_id"],
                        stage="formalizer",
                        attempts=attempt_num,
                        success=False,
                        verified=False,
                        lean_code=lean_code,
                        error_message=str(error_msg),
                        raw_output=output,
                        attempt_history=history,
                    )
                )

        return finished
