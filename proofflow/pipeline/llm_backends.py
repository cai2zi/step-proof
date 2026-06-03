from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from ..llm_worker import LLMWorkerConfig, LLMWorkerPool
from .specs import ModelSpec


Message = Dict[str, str]
Generation = Dict[str, Any]


class LLMBackend(Protocol):
    def generate_many(self, message_batches: List[List[Message]]) -> List[Generation]:
        ...

    def close(self, force: bool = False) -> None:
        ...


@dataclass
class FakeLLMBackend:
    """Deterministic backend for characterization and smoke tests."""

    text: str = "```lean4\nexample : True := by trivial\n```"

    def generate_many(self, message_batches: List[List[Message]]) -> List[Generation]:
        return [
            {
                "text": self.text,
                "prompt_token_overflow": False,
                "output_truncated": False,
                "finish_reason": "fake",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            for _ in message_batches
        ]

    def close(self, force: bool = False) -> None:
        return None


class VLLMBackend:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.pool = LLMWorkerPool(
            base_config=LLMWorkerConfig(
                name=spec.name,
                gpus="",
                model_path=spec.model_path,
                tensor_parallel_size=spec.tensor_parallel_size,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                token_limit=spec.token_limit,
                dtype=spec.dtype,
                gpu_memory_utilization=spec.gpu_memory_utilization,
                top_p=spec.top_p,
                presence_penalty=spec.presence_penalty,
                frequency_penalty=spec.frequency_penalty,
                seed=spec.seed,
                top_k=spec.top_k,
                chat_template_kwargs=dict(spec.chat_template_kwargs),
            ),
            instances=spec.instances,
            gpus=spec.gpus,
            startup_timeout=spec.startup_timeout,
            parallel_startup=spec.parallel_startup,
            startup_stagger_seconds=spec.startup_stagger_seconds,
        )
        self._next_worker = 0

    def generate_many(self, message_batches: List[List[Message]]) -> List[Generation]:
        if not message_batches:
            return []
        worker = self.pool[self._next_worker]
        self._next_worker = (self._next_worker + 1) % max(1, len(self.pool))
        return worker.generate_with_metadata(message_batches)

    def close(self, force: bool = False) -> None:
        self.pool.close(force=force)


class OpenAIChatBackend:
    """OpenAI-compatible chat backend with local prompt token preflight."""

    def __init__(self, spec: ModelSpec, *, api_key: str) -> None:
        try:
            import httpx  # noqa: F401
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenAIChatBackend requires openai and httpx.") from exc

        self.spec = spec
        self.api_key = api_key
        self._openai_cls = OpenAI
        self._local = threading.local()
        self._encoding = self._load_tiktoken_encoding(spec.name)
        self._tokenizer = self._load_tokenizer(spec.api_tokenizer_path)
        if self._tokenizer is not None:
            self.token_counter = f"transformers:{spec.api_tokenizer_path}"
        elif self._encoding is not None:
            self.token_counter = "tiktoken"
        else:
            self.token_counter = "char_estimate"

    @staticmethod
    def _load_tiktoken_encoding(model: str) -> Any:
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _load_tokenizer(path: str) -> Any:
        if not path:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("--api-tokenizer-path requires transformers.") from exc
        return AutoTokenizer.from_pretrained(path)

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._openai_cls(
                base_url=self.spec.api_base_url,
                api_key=self.api_key,
                timeout=self.spec.api_timeout,
                max_retries=0,
            )
            self._local.client = client
        return client

    def _message_token_count(self, messages: List[Message]) -> int:
        if self._tokenizer is not None:
            try:
                return len(
                    self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        **dict(self.spec.chat_template_kwargs),
                    )
                )
            except Exception:
                joined = "\n".join(
                    f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages
                )
                return len(self._tokenizer.encode(joined, add_special_tokens=False))
        if self._encoding is None:
            return sum(max(1, len(str(msg.get("content", ""))) // 4) for msg in messages)
        return sum(len(self._encoding.encode(str(msg.get("content", "")))) for msg in messages)

    def _request_kwargs(self, messages: List[Message], token_param: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.spec.name,
            "messages": messages,
            token_param: self.spec.max_tokens,
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
        }
        if self.spec.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.spec.presence_penalty
        if self.spec.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.spec.frequency_penalty
        return kwargs

    def _call_once(self, messages: List[Message]) -> Generation:
        request_started = time.perf_counter()
        token_param = "max_tokens"
        fallback_seconds = 0.0
        try:
            completion = self._client().chat.completions.create(
                **self._request_kwargs(messages, token_param)
            )
        except Exception as exc:
            text = str(exc).lower()
            if "max_tokens" not in text or "max_completion_tokens" not in text:
                raise
            fallback_started = time.perf_counter()
            token_param = "max_completion_tokens"
            completion = self._client().chat.completions.create(
                **self._request_kwargs(messages, token_param)
            )
            fallback_seconds = time.perf_counter() - fallback_started

        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        usage = getattr(completion, "usage", None)
        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        return {
            "text": getattr(message, "content", None) or "",
            "prompt_token_overflow": False,
            "output_truncated": "length" in str(finish_reason or "").lower(),
            "finish_reason": None if finish_reason is None else str(finish_reason),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None)
            if usage is not None
            else None,
            "api_model": self.spec.name,
            "api_base_url": self.spec.api_base_url,
            "api_token_param": token_param,
            "api_request_seconds": time.perf_counter() - request_started,
            "api_fallback_seconds": fallback_seconds,
        }

    def generate_many(self, message_batches: List[List[Message]]) -> List[Generation]:
        results: List[Generation] = []
        token_limit = self.spec.api_input_token_limit
        if token_limit < 0:
            token_limit = self.spec.token_limit
        for messages in message_batches:
            total_started = time.perf_counter()
            prompt_tokens = self._message_token_count(messages)
            if token_limit > 0 and prompt_tokens > token_limit:
                results.append(
                    {
                        "text": None,
                        "prompt_token_overflow": True,
                        "output_truncated": False,
                        "finish_reason": None,
                        "prompt_tokens": prompt_tokens,
                        "token_limit": token_limit,
                        "token_counter": self.token_counter,
                    }
                )
                continue
            attempts = 0
            sleep_seconds = 0.0
            errors: List[Dict[str, Any]] = []
            for attempt in range(self.spec.api_max_retries + 1):
                attempts += 1
                try:
                    result = self._call_once(messages)
                    if result.get("prompt_tokens") is None:
                        result["prompt_tokens"] = prompt_tokens
                    result["token_counter"] = self.token_counter
                    result["api_attempts"] = attempts
                    result["api_sleep_seconds"] = sleep_seconds
                    result["api_total_seconds"] = time.perf_counter() - total_started
                    result["response_chars"] = len(str(result.get("text") or ""))
                    results.append(result)
                    break
                except Exception as exc:
                    errors.append(
                        {
                            "attempt": attempts,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if attempt >= self.spec.api_max_retries:
                        results.append(
                            {
                                "text": "",
                                "prompt_token_overflow": False,
                                "output_truncated": False,
                                "finish_reason": "api_error",
                                "prompt_tokens": prompt_tokens,
                                "token_counter": self.token_counter,
                                "api_error": str(exc),
                                "api_error_type": type(exc).__name__,
                                "api_error_history": errors,
                                "api_attempts": attempts,
                                "api_sleep_seconds": sleep_seconds,
                                "api_total_seconds": time.perf_counter() - total_started,
                            }
                        )
                        break
                    sleep_for = self.spec.api_retry_sleep * (attempt + 1)
                    time.sleep(sleep_for)
                    sleep_seconds += sleep_for
        return results

    def close(self, force: bool = False) -> None:
        client = getattr(self._local, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def build_llm_backend(spec: ModelSpec, *, api_key: Optional[str] = None) -> LLMBackend:
    backend = spec.backend.strip().lower()
    if backend == "fake":
        return FakeLLMBackend()
    if backend == "vllm":
        return VLLMBackend(spec)
    if backend == "api":
        if not api_key:
            raise RuntimeError("API backend requires an API key.")
        return OpenAIChatBackend(spec, api_key=api_key)
    raise ValueError(f"Unknown LLM backend: {spec.backend!r}")
