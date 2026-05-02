from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import asdict, dataclass, field
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
            if req_type != "generate":
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
    ) -> None:
        self.config = config
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
        self._wait_until_ready(startup_timeout)

    def _wait_until_ready(self, timeout: int) -> None:
        try:
            message = self._response_queue.get(timeout=timeout)
        except queue.Empty as e:
            self.close(force=True)
            raise RuntimeError(
                f"{self.config.name} worker did not become ready within {timeout} seconds."
            ) from e

        msg_type = message.get("type")
        if msg_type == "ready":
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
