from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMWorkerConfig:
    name: str
    gpus: str
    model_path: str
    tensor_parallel_size: int
    max_tokens: int
    temperature: float
    token_limit: int
    dtype: str
    gpu_memory_utilization: float
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 42
    top_k: int = 20
    chat_template_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )


def _worker_main(
    config_dict: Dict[str, object],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    try:
        config = LLMWorkerConfig(**config_dict)
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

        from .local_vllm import LocalLLMManager

        manager = LocalLLMManager(
            model_path=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            seed=config.seed,
            top_k=config.top_k,
            token_limit=config.token_limit,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        response_queue.put(
            {
                "type": "ready",
                "name": config.name,
                "gpus": config.gpus,
            }
        )

        while True:
            request = request_queue.get()
            req_type = request.get("type")
            if req_type == "shutdown":
                return
            if req_type not in {"generate", "generate_with_metadata"}:
                response_queue.put(
                    {
                        "type": "error",
                        "error": f"Unknown request type: {req_type}",
                    }
                )
                continue
            try:
                if os.getenv("STEP_PROOF_RL_TRACE", "1") != "0":
                    print(
                        f"[{config.name}] vLLM generate start "
                        f"batch_size={len(request['message_batches'])} "
                        f"gpus={config.gpus} tp={config.tensor_parallel_size} "
                        f"max_tokens={config.max_tokens}",
                        flush=True,
                    )
                if req_type == "generate_with_metadata":
                    outputs = manager.batch_generate_with_metadata(request["message_batches"])
                else:
                    outputs = manager.batch_generate(request["message_batches"])
                if os.getenv("STEP_PROOF_RL_TRACE", "1") != "0":
                    print(
                        f"[{config.name}] vLLM generate done "
                        f"batch_size={len(request['message_batches'])}",
                        flush=True,
                    )
                response_queue.put({"type": "result", "outputs": outputs})
            except Exception:
                response_queue.put(
                    {
                        "type": "error",
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        response_queue.put(
            {
                "type": "startup_error",
                "error": traceback.format_exc(),
            }
        )


class LLMWorkerClient:
    """Owns a dedicated subprocess that keeps one vLLM model resident on a fixed GPU set."""

    def __init__(
        self,
        config: LLMWorkerConfig,
        startup_timeout: int = 1800,
        wait_ready: bool = True,
    ) -> None:
        self.config = config
        self._ready = False
        self._ctx = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue()
        self._response_queue: mp.Queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(asdict(config), self._request_queue, self._response_queue),
            daemon=False,
            name=f"llm-worker-{config.name}",
        )
        self._process.start()
        if wait_ready:
            self.wait_until_ready(startup_timeout)

    def wait_until_ready(self, timeout: int) -> None:
        if self._ready:
            return
        try:
            message = self._response_queue.get(timeout=timeout)
        except queue.Empty as e:
            self.close(force=True)
            raise RuntimeError(
                f"{self.config.name} worker did not become ready within {timeout} seconds."
            ) from e

        msg_type = message.get("type")
        if msg_type == "ready":
            self._ready = True
            return

        self.close(force=True)
        raise RuntimeError(
            f"{self.config.name} worker failed during startup:\n{message.get('error', message)}"
        )

    def generate(
        self,
        message_batches: List[List[Dict[str, str]]],
        timeout: Optional[int] = None,
    ) -> List[Optional[str]]:
        self._request_queue.put(
            {
                "type": "generate",
                "message_batches": message_batches,
            }
        )
        try:
            response = self._response_queue.get(timeout=timeout) if timeout else self._response_queue.get()
        except queue.Empty as e:
            raise RuntimeError(
                f"{self.config.name} worker timed out waiting for generation results."
            ) from e

        msg_type = response.get("type")
        if msg_type == "result":
            return response["outputs"]
        raise RuntimeError(
            f"{self.config.name} worker generation failed:\n{response.get('error', response)}"
        )

    def generate_with_metadata(
        self,
        message_batches: List[List[Dict[str, str]]],
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._request_queue.put(
            {
                "type": "generate_with_metadata",
                "message_batches": message_batches,
            }
        )
        try:
            response = self._response_queue.get(timeout=timeout) if timeout else self._response_queue.get()
        except queue.Empty as e:
            raise RuntimeError(
                f"{self.config.name} worker timed out waiting for generation results."
            ) from e

        msg_type = response.get("type")
        if msg_type == "result":
            return response["outputs"]
        raise RuntimeError(
            f"{self.config.name} worker generation failed:\n{response.get('error', response)}"
        )

    async def generate_async(
        self,
        message_batches: List[List[Dict[str, str]]],
        timeout: Optional[int] = None,
    ) -> List[Optional[str]]:
        return await asyncio.to_thread(self.generate, message_batches, timeout)

    async def generate_with_metadata_async(
        self,
        message_batches: List[List[Dict[str, str]]],
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.generate_with_metadata, message_batches, timeout)

    def close(self, force: bool = False) -> None:
        if not hasattr(self, "_process") or self._process is None:
            return
        if self._process.is_alive() and not force:
            try:
                self._request_queue.put({"type": "shutdown"})
                self._process.join(timeout=5)
            except Exception:
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


def split_gpu_groups(gpus: str, *, instances: int, tensor_parallel_size: int) -> List[str]:
    devices = [device.strip() for device in gpus.split(",") if device.strip()]
    if instances < 1:
        raise ValueError(f"instances must be >= 1, got {instances}")
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
    total_needed = instances * tensor_parallel_size
    if len(devices) < total_needed:
        raise ValueError(
            f"Not enough GPUs for {instances} worker(s) with tp={tensor_parallel_size}. "
            f"Need {total_needed}, got {len(devices)} from {gpus!r}."
        )
    return [
        ",".join(devices[start : start + tensor_parallel_size])
        for start in range(0, total_needed, tensor_parallel_size)
    ]


class LLMWorkerPool:
    """Owns multiple independent vLLM worker subprocesses."""

    def __init__(
        self,
        *,
        base_config: LLMWorkerConfig,
        instances: int,
        gpus: str,
        startup_timeout: int = 1800,
        parallel_startup: bool = False,
        startup_stagger_seconds: float = 0.0,
    ) -> None:
        self.gpu_groups = split_gpu_groups(
            gpus,
            instances=instances,
            tensor_parallel_size=base_config.tensor_parallel_size,
        )
        self.workers: List[LLMWorkerClient] = []
        try:
            for idx, gpu_group in enumerate(self.gpu_groups):
                config = replace(
                    base_config,
                    name=f"{base_config.name}-{idx}",
                    gpus=gpu_group,
                )
                self.workers.append(
                    LLMWorkerClient(
                        config=config,
                        startup_timeout=startup_timeout,
                        wait_ready=not parallel_startup,
                    )
                )
                if parallel_startup:
                    print(
                        f"[init] started {config.name} on gpus={gpu_group}",
                        flush=True,
                    )
                    if startup_stagger_seconds > 0 and idx + 1 < len(self.gpu_groups):
                        time.sleep(startup_stagger_seconds)
            if parallel_startup:
                for idx, worker in enumerate(self.workers):
                    worker.wait_until_ready(startup_timeout)
                    print(
                        f"[init] {worker.config.name} ready on gpus={worker.config.gpus}",
                        flush=True,
                    )
        except Exception:
            self.close(force=True)
            raise

    def __len__(self) -> int:
        return len(self.workers)

    def __getitem__(self, idx: int) -> LLMWorkerClient:
        return self.workers[idx]

    def close(self, force: bool = False) -> None:
        for worker in self.workers:
            worker.close(force=force)
        self.workers.clear()
