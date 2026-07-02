from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import httpx
except ImportError:  # pragma: no cover - only needed by the API backend.
    httpx = None  # type: ignore[assignment]

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional local preflight helper.
    tiktoken = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - only needed by the API backend.
    OpenAI = None  # type: ignore[assignment]

from .fdg_graph import FDG_OUTPUT_TRUNCATED_RETRY_HINT
from .fdg_stage_common import (
    FORM_TERMINAL,
    FORMALIZER_CONTEXT_MODES,
    build_fdg_form_messages,
    fdg_fact_should_execute,
    fdg_stage2_checkpoint_payload,
    fdg_stage2_final_payload,
    fdg_stage2_record_terminal,
    fresh_fdg_stage2_record_state,
    restore_fdg_stage2_record_state,
)
from .lean_check import LeanServer
from .llm_worker import LLMWorkerConfig, LLMWorkerPool
from .runtime_common import (
    append_jsonl,
    extract_last_lean_block,
    load_jsonl,
    utc_now_iso,
    write_json_atomic,
)

load_dotenv()

DEFAULT_FORMALIZER_MODEL_PATH = os.getenv(
    "FORMALIZER_MODEL_PATH", "/data/czx/models/Goedel-Formalizer-V2-8B"
)
DEFAULT_GPUS = os.getenv("STAGE2_GPUS", os.getenv("GRAPH_GPUS", "0,1,2,3"))
DEFAULT_FORMALIZER_GPUS = os.getenv("FORMALIZER_GPUS", "")
DEFAULT_FORMALIZER_TP = int(os.getenv("FORMALIZER_TP", "2"))
DEFAULT_MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")
DEFAULT_LEAN_BACKEND = os.getenv("LEAN_BACKEND", "subprocess")
DEFAULT_LEAN_API_URL = os.getenv("KIMINA_API_URL", os.getenv("LEAN_SERVER_API_URL", "http://localhost:8000"))
DEFAULT_LEAN_API_KEY_ENV = os.getenv("LEAN_API_KEY_ENV", "KIMINA_API_KEY")
DEFAULT_LEAN_SERVER_TIMEOUT = int(os.getenv("LEAN_SERVER_TIMEOUT", "300"))
DEFAULT_API_BASE_URL = os.getenv("FORMALIZER_API_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_MODEL = os.getenv("FORMALIZER_API_MODEL", "gpt-4.1")
DEFAULT_API_KEY_ENV = os.getenv("FORMALIZER_API_KEY_ENV", "OPENAI_API_KEY")


class Stage2APIClient:
    """OpenAI-compatible chat API backend for Stage 2 formalization."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        input_token_limit: int,
        tokenizer_path: str,
        tokenizer_chat_template_kwargs: Optional[Dict[str, Any]],
        timeout: float,
        api_max_retries: int,
        retry_sleep: float,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.input_token_limit = input_token_limit
        self.tokenizer_path = tokenizer_path
        self.tokenizer_chat_template_kwargs = dict(tokenizer_chat_template_kwargs or {})
        self.timeout = timeout
        self.api_max_retries = max(0, api_max_retries)
        self.retry_sleep = max(0.0, retry_sleep)
        self._local = threading.local()
        if OpenAI is None or httpx is None:
            raise RuntimeError(
                "The API backend requires the openai and httpx packages. "
                "Install project dependencies or run with --backend vllm."
            )
        if tiktoken is None:
            self._encoding = None
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        self._tokenizer = self._load_tokenizer(tokenizer_path)
        if self._tokenizer is not None:
            self.token_counter = f"transformers:{tokenizer_path}"
        elif self._encoding is not None:
            self.token_counter = "tiktoken"
        else:
            self.token_counter = "char_estimate"

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        if not tokenizer_path:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "--api-tokenizer-path requires the transformers package. "
                "Install transformers or omit --api-tokenizer-path."
            ) from exc
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path!r}: {exc}") from exc

    def _client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
            self._local.client = client
        return client

    def _message_token_count(self, messages: List[Dict[str, str]]) -> int:
        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.tokenizer_chat_template_kwargs,
                )
                return len(tokens)
            except Exception:
                joined = "\n".join(
                    f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages
                )
                return len(self._tokenizer.encode(joined, add_special_tokens=False))

        total = 0
        for msg in messages:
            content = str(msg.get("content", ""))
            if self._encoding is None:
                total += max(1, len(content) // 4)
            else:
                total += len(self._encoding.encode(content))
        return total

    def _request_kwargs(self, messages: List[Dict[str, str]], token_param: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            token_param: self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.frequency_penalty
        return kwargs

    @staticmethod
    def _should_retry_with_max_completion_tokens(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "max_tokens" in text
            and "max_completion_tokens" in text
            and ("unsupported" in text or "not supported" in text or "invalid" in text)
        )

    def _call_once(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        client = self._client()
        token_param = "max_tokens"
        request_started = time.perf_counter()
        fallback_seconds = 0.0
        try:
            completion = client.chat.completions.create(
                **self._request_kwargs(messages, token_param)
            )
        except Exception as exc:
            if not self._should_retry_with_max_completion_tokens(exc):
                raise
            fallback_started = time.perf_counter()
            token_param = "max_completion_tokens"
            completion = client.chat.completions.create(
                **self._request_kwargs(messages, token_param)
            )
            fallback_seconds = time.perf_counter() - fallback_started

        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        reason_text = str(finish_reason or "").lower()
        return {
            "text": content or "",
            "prompt_token_overflow": False,
            "output_truncated": "length" in reason_text,
            "finish_reason": None if finish_reason is None else str(finish_reason),
            "stop_reason": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_model": self.model,
            "api_base_url": self.base_url,
            "api_token_param": token_param,
            "api_request_seconds": time.perf_counter() - request_started,
            "api_fallback_seconds": fallback_seconds,
        }

    def generate_with_metadata(
        self,
        message_batches: List[List[Dict[str, str]]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for messages in message_batches:
            total_started = time.perf_counter()
            token_started = time.perf_counter()
            prompt_tokens = self._message_token_count(messages)
            token_count_seconds = time.perf_counter() - token_started
            if self.input_token_limit > 0 and prompt_tokens > self.input_token_limit:
                results.append(
                    {
                        "text": None,
                        "prompt_token_overflow": True,
                        "output_truncated": False,
                        "finish_reason": None,
                        "stop_reason": None,
                        "prompt_tokens": prompt_tokens,
                        "token_limit": self.input_token_limit,
                        "token_counter": self.token_counter,
                        "token_count_seconds": token_count_seconds,
                        "api_attempts": 0,
                        "api_request_seconds": 0.0,
                        "api_sleep_seconds": 0.0,
                        "api_total_seconds": time.perf_counter() - total_started,
                    }
                )
                continue

            attempts = 0
            request_seconds = 0.0
            sleep_seconds = 0.0
            error_history: List[Dict[str, Any]] = []
            for attempt in range(self.api_max_retries + 1):
                attempts += 1
                attempt_started = time.perf_counter()
                try:
                    result = self._call_once(messages)
                    request_seconds += float(result.get("api_request_seconds") or 0.0)
                    if result.get("prompt_tokens") is None:
                        result["prompt_tokens"] = prompt_tokens
                    result["token_counter"] = self.token_counter
                    result["token_count_seconds"] = token_count_seconds
                    result["api_attempts"] = attempts
                    result["api_request_seconds"] = request_seconds
                    result["api_sleep_seconds"] = sleep_seconds
                    result["api_total_seconds"] = time.perf_counter() - total_started
                    result["response_chars"] = len(str(result.get("text") or ""))
                    results.append(result)
                    break
                except Exception as exc:
                    attempt_seconds = time.perf_counter() - attempt_started
                    request_seconds += attempt_seconds
                    error_history.append(
                        {
                            "attempt": attempts,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "elapsed_seconds": attempt_seconds,
                        }
                    )
                    if attempt >= self.api_max_retries:
                        results.append(
                            {
                                "text": "",
                                "prompt_token_overflow": False,
                                "output_truncated": False,
                                "finish_reason": "api_error",
                                "stop_reason": None,
                                "prompt_tokens": prompt_tokens,
                                "token_counter": self.token_counter,
                                "api_error": str(exc),
                                "api_error_type": type(exc).__name__,
                                "api_error_history": error_history,
                                "token_count_seconds": token_count_seconds,
                                "api_attempts": attempts,
                                "api_request_seconds": request_seconds,
                                "api_sleep_seconds": sleep_seconds,
                                "api_total_seconds": time.perf_counter() - total_started,
                            }
                        )
                        break
                    sleep_for = self.retry_sleep * (attempt + 1)
                    time.sleep(sleep_for)
                    sleep_seconds += sleep_for
        return results

    def close(self, force: bool = False) -> None:
        client = getattr(self._local, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


class FDGStage2Runner:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        lean_server: Optional[LeanServer] = None,
        owned_lean_server: Optional[bool] = None,
    ) -> None:
        self.args = args
        self.records: Dict[str, Dict[str, Any]] = {}
        self.form_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.state_lock = asyncio.Lock()
        self.lean_semaphore = asyncio.Semaphore(args.lean_check_concurrency)
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir
        self.done_ids: set[str] = set()
        self.formalizers: Optional[Any] = None
        self.lean_server: Optional[LeanServer] = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.validation_backpressure_limit = max(1, args.max_pending_validation_batches) * max(1, args.form_batch_size)
        self.validation_backpressure = asyncio.Semaphore(self.validation_backpressure_limit)
        self.validation_queue: asyncio.Queue[Tuple[Dict[str, Any], Dict[str, Any]]] = asyncio.Queue()
        self.validation_workers: List[asyncio.Task] = []
        self.running_validation_items = 0
        self.validation_error: Optional[BaseException] = None
        self.stage_started_perf: Optional[float] = None
        self.gpu_idle_wait_seconds = 0.0
        self.lean_compile_seconds = 0.0
        self.lean_compile_count = 0

    def _pending_validation_items(self) -> int:
        return self.validation_queue.qsize() + self.running_validation_items

    def _effective_form_batch_size(self) -> int:
        if self.args.backend == "api":
            return 1
        return self.args.form_batch_size

    def _current_prompt_meta(self) -> Tuple[str, str]:
        prompt = str(
            getattr(self.args, "formalizer_prompt", "formalize_obligation")
            or "formalize_obligation"
        )
        context_mode = str(
            getattr(self.args, "formalizer_context_mode", "c0_parent") or "c0_parent"
        )
        return prompt, context_mode

    def _validate_existing_prompt_meta(
        self,
        meta: Dict[str, Any],
        *,
        record_id: str,
        source: str,
    ) -> None:
        prompt, context_mode = self._current_prompt_meta()
        existing_prompt = str(meta.get("formalizer_prompt") or "")
        if existing_prompt and existing_prompt != prompt:
            raise RuntimeError(
                f"Record {record_id} {source} was created with "
                f"formalizer_prompt={existing_prompt!r}, but current config uses {prompt!r}. "
                "Use a new exp.name or disable resume."
            )
        existing_context_mode = str(meta.get("formalizer_context_mode") or "")
        if existing_context_mode:
            if existing_context_mode != context_mode:
                raise RuntimeError(
                    f"Record {record_id} {source} was created with "
                    f"formalizer_context_mode={existing_context_mode!r}, but current config uses "
                    f"{context_mode!r}. Use a new exp.name or disable resume."
                )
        elif context_mode != "c0_parent":
            raise RuntimeError(
                f"Record {record_id} {source} has no formalizer_context_mode metadata and is only "
                "compatible with c0_parent. Use a new exp.name or disable resume."
            )

    def _load_done_ids_checked(self) -> set[str]:
        done_ids: set[str] = set()
        for row in load_jsonl(self.out_path):
            meta = dict(row.get("meta") or {})
            record_id = str(meta.get("record_id") or "").strip()
            if not record_id:
                continue
            self._validate_existing_prompt_meta(
                meta,
                record_id=record_id,
                source="completed result",
            )
            done_ids.add(record_id)
        return done_ids

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else self._load_done_ids_checked()
        source_rows = load_jsonl(self.args.infile)
        loaded = resumed_count = partial_count = form_done = empty_skipped = 0

        for row in source_rows:
            record_id = str(row.get("meta", {}).get("record_id", "")).strip()
            if not record_id:
                continue
            if record_id in self.done_ids:
                resumed_count += 1
                continue

            graph_facts = (row.get("graph") or {}).get("facts", [])
            if not graph_facts:
                empty_skipped += 1
                append_jsonl(
                    self.failed_path,
                    {
                        "meta": row.get("meta", {}),
                        "error": "Empty FDG graph (no facts). Skipped by Stage 2.",
                        "created_at": utc_now_iso(),
                    },
                )
                continue

            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if not self.args.no_resume and ckpt_path.is_file():
                record = restore_fdg_stage2_record_state(
                    json.loads(ckpt_path.read_text(encoding="utf-8"))
                )
                self._ensure_prompt_meta(record, resumed=True)
                partial_count += 1
                for fact in record["facts"].values():
                    if fact["form_status"] in FORM_TERMINAL:
                        form_done += 1
            else:
                record = fresh_fdg_stage2_record_state(row)
                self._ensure_prompt_meta(record, resumed=False)

            self.records[record_id] = record
            loaded += 1
            if self.args.limit >= 0 and loaded >= self.args.limit:
                break

        print(f"\n[resume] Fully completed records skipped: {resumed_count}")
        print(f"[resume] Empty FDG records skipped: {empty_skipped}")
        print(f"[resume] Partially completed records loaded: {partial_count}")
        print(f"[resume] Facts already formalized/skipped: {form_done}")
        print(f"[resume] Total pending records to process: {loaded}\n")

    def _ensure_prompt_meta(self, record: Dict[str, Any], *, resumed: bool) -> None:
        prompt, context_mode = self._current_prompt_meta()
        meta = record.setdefault("meta", {})
        record_id = str(meta.get("record_id") or "<unknown>")
        if resumed:
            self._validate_existing_prompt_meta(
                meta,
                record_id=record_id,
                source="checkpoint",
            )
        meta["formalizer_prompt"] = prompt
        meta["formalizer_context_mode"] = context_mode

    def _resolve_formalizer_gpus(self) -> str:
        if self.args.formalizer_gpus:
            return self.args.formalizer_gpus
        all_devices = [device.strip() for device in self.args.gpus.split(",") if device.strip()]
        total_needed = self.args.formalizer_instances * self.args.formalizer_tensor_parallel_size
        if len(all_devices) < total_needed:
            raise RuntimeError(
                "Not enough GPUs in --gpus to derive the formalizer worker. "
                f"Need {total_needed}, got {len(all_devices)} from {self.args.gpus!r}."
            )
        return ",".join(all_devices[:total_needed])

    async def init_runtime(self) -> None:
        chat_template_kwargs: Dict[str, Any] = {"enable_thinking": False}
        if self.args.formalizer_chat_template_kwargs_json:
            parsed = json.loads(self.args.formalizer_chat_template_kwargs_json)
            if not isinstance(parsed, dict):
                raise RuntimeError("--formalizer-chat-template-kwargs-json must be a JSON object")
            chat_template_kwargs = parsed

        if self.args.backend == "vllm":
            formalizer_gpus = self._resolve_formalizer_gpus()
            print(
                "[init] loading formalizer",
                self.args.formalizer_model_path,
                (
                    f"(backend=vllm, instances={self.args.formalizer_instances}, "
                    f"tp={self.args.formalizer_tensor_parallel_size}, gpus={formalizer_gpus}) ..."
                ),
            )
            self.formalizers = LLMWorkerPool(
                base_config=LLMWorkerConfig(
                    name="formalizer",
                    gpus="",
                    model_path=self.args.formalizer_model_path or DEFAULT_FORMALIZER_MODEL_PATH,
                    tensor_parallel_size=self.args.formalizer_tensor_parallel_size,
                    max_tokens=self.args.formalizer_max_tokens,
                    temperature=self.args.formalizer_temperature,
                    token_limit=self.args.formalizer_token_limit,
                    dtype=self.args.dtype,
                    gpu_memory_utilization=self.args.gpu_memory_utilization,
                    top_p=self.args.formalizer_top_p,
                    presence_penalty=self.args.formalizer_presence_penalty,
                    frequency_penalty=self.args.formalizer_frequency_penalty,
                    seed=self.args.formalizer_seed,
                    top_k=self.args.formalizer_top_k,
                    chat_template_kwargs=chat_template_kwargs,
                ),
                instances=self.args.formalizer_instances,
                gpus=formalizer_gpus,
                startup_timeout=self.args.formalizer_startup_timeout,
                parallel_startup=self.args.formalizer_parallel_startup,
                startup_stagger_seconds=self.args.formalizer_startup_stagger_seconds,
            )
        else:
            if self.args.api_concurrency < 1:
                raise RuntimeError("--api-concurrency must be >= 1")
            api_key = os.getenv(self.args.api_key_env)
            if not api_key:
                raise RuntimeError(f"--api-key-env={self.args.api_key_env!r} is not set")
            input_token_limit = self.args.api_input_token_limit
            if input_token_limit < 0:
                input_token_limit = self.args.formalizer_token_limit
            print(
                f"[init] using API formalizer backend model={self.args.api_model} "
                f"base_url={self.args.api_base_url} concurrency={self.args.api_concurrency} "
                f"input_token_limit={input_token_limit} "
                f"api_tokenizer_path={self.args.api_tokenizer_path or '<tiktoken>'}"
            )
            self.formalizers = [
                Stage2APIClient(
                    model=self.args.api_model,
                    base_url=self.args.api_base_url,
                    api_key=api_key,
                    max_tokens=self.args.formalizer_max_tokens,
                    temperature=self.args.formalizer_temperature,
                    top_p=self.args.formalizer_top_p,
                    presence_penalty=self.args.formalizer_presence_penalty,
                    frequency_penalty=self.args.formalizer_frequency_penalty,
                    input_token_limit=input_token_limit,
                    tokenizer_path=self.args.api_tokenizer_path,
                    tokenizer_chat_template_kwargs=chat_template_kwargs,
                    timeout=self.args.api_timeout,
                    api_max_retries=self.args.api_max_retries,
                    retry_sleep=self.args.api_retry_sleep,
                )
                for _ in range(self.args.api_concurrency)
            ]
        if self.lean_server is None:
            if self.args.lean_backend == "kimina_server":
                self.lean_server = LeanServer(
                    api_url=self.args.lean_api_url,
                    backend=self.args.lean_backend,
                    api_key_env=self.args.lean_api_key_env,
                    server_timeout=self.args.lean_server_timeout,
                    server_reuse=self.args.lean_server_reuse,
                    server_debug=self.args.lean_server_debug,
                )
            else:
                pool_size = (
                    self.args.lean_worker_pool_size
                    if self.args.lean_worker_pool_size > 0
                    else self.args.lean_check_concurrency
                )
                self.lean_server = LeanServer(
                    project_path=self.args.mathlib_path or DEFAULT_MATHLIB_PATH,
                    backend=self.args.lean_backend,
                    pool_size=pool_size,
                    temp_root=str(self.args.lean_temp_dir),
                )
        else:
            print("[init] fdg stage2 using shared Lean runtime.")
        print("[init] fdg stage2 runtime ready.\n")

    async def seed_ready_queue(self) -> None:
        async with self.state_lock:
            for record_id, record in self.records.items():
                if fdg_stage2_record_terminal(record):
                    await self._persist_record_locked(record_id)
                    continue
                for fact_id, fact in record["facts"].items():
                    if fact["form_status"] == "pending":
                        await self._enqueue_form_locked(record_id, fact_id)

    async def _persist_record_locked(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if fdg_stage2_record_terminal(record):
            payload = fdg_stage2_final_payload(record)
            append_jsonl(self.out_path, payload)
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            record["record_status"] = payload["execution"]["record_status"]
            self.done_ids.add(record_id)
            return
        write_json_atomic(self.checkpoint_dir / f"{record_id}.json", fdg_stage2_checkpoint_payload(record))

    async def _enqueue_form_locked(self, record_id: str, fact_id: str) -> None:
        fact = self.records[record_id]["facts"][fact_id]
        if fact.get("_form_enqueued"):
            return
        fact["_form_enqueued"] = True
        await self.form_queue.put((record_id, fact_id))

    async def _validate_lean(self, lean_code: str, job_id: str) -> Tuple[bool, bool, Any]:
        assert self.lean_server is not None
        async with self.lean_semaphore:
            started_perf = time.perf_counter()
            result = await self.lean_server.check_lean_string_async(
                lean_code,
                temp_root=str(self.args.lean_temp_dir),
                job_id=job_id,
            )
        self.lean_compile_seconds += time.perf_counter() - started_perf
        self.lean_compile_count += 1
        return result

    async def _validate_batch_output(self, task: Dict[str, Any], generation: Dict[str, Any]) -> Dict[str, Any]:
        if generation.get("prompt_token_overflow"):
            return {"kind": "token_overflow", "error_msg": "token_overflow", "lean_code": ""}
        if generation.get("api_error"):
            return {
                "kind": "api_error",
                "error_msg": generation.get("api_error") or "api_error",
                "lean_code": "",
                "api_error_type": generation.get("api_error_type"),
            }
        output = generation.get("text") or ""
        if generation.get("output_truncated"):
            return {
                "kind": "output_truncated",
                "error_msg": FDG_OUTPUT_TRUNCATED_RETRY_HINT,
                "lean_code": "",
                "finish_reason": generation.get("finish_reason"),
                "stop_reason": generation.get("stop_reason"),
            }
        try:
            lean_code = extract_last_lean_block(output)
        except ValueError as exc:
            return {"kind": "extract_error", "error_msg": str(exc), "lean_code": ""}
        lean_pass, _lean_verify, error_msg = await self._validate_lean(
            lean_code,
            job_id=f"fdg_stage2_{task['record_id']}_{task['fact_id']}_{task['attempt_num']}",
        )
        return {
            "kind": "validated",
            "lean_code": lean_code,
            "lean_pass": lean_pass,
            "error_msg": error_msg,
        }

    async def _pop_batch(self, batch_size: int) -> List[Tuple[str, str]]:
        timeout = max(self.args.batch_wait_ms, 1) / 1000.0
        wait_started_perf = time.perf_counter()
        try:
            first_item = await asyncio.wait_for(self.form_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
            return []
        self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
        items = [first_item]
        while len(items) < batch_size:
            try:
                items.append(self.form_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def _prepare_batch(self, batch_items: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        async with self.state_lock:
            for record_id, fact_id in batch_items:
                record = self.records.get(record_id)
                if record is None:
                    continue
                fact = record["facts"].get(fact_id)
                if fact is None:
                    continue
                fact["_form_enqueued"] = False
                if not fdg_fact_should_execute(fact):
                    fact["form_status"] = "skipped"
                    continue
                if fact["form_status"] != "pending":
                    continue
                fact["form_status"] = "running"
                if not fact["form_messages"]:
                    fact["form_messages"] = build_fdg_form_messages(
                        fact,
                        record=record,
                        context_mode=self.args.formalizer_context_mode,
                        prompt_name=self.args.formalizer_prompt,
                    )
                attempt_num = int(fact.get("form_retries_used", 0)) + 1
                tasks.append(
                    {
                        "record_id": record_id,
                        "fact_id": fact_id,
                        "messages": copy.deepcopy(fact["form_messages"]),
                        "attempt_num": attempt_num,
                    }
                )
        return tasks

    async def _generate_outputs(self, worker_id: int, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assert self.formalizers is not None
        return await asyncio.to_thread(
            self.formalizers[worker_id].generate_with_metadata,
            [task["messages"] for task in tasks],
        )

    async def _validate_and_apply_one_output(self, task: Dict[str, Any], generation: Dict[str, Any]) -> None:
        result = await self._validate_batch_output(task, generation)
        async with self.state_lock:
            record = self.records[task["record_id"]]
            fact = record["facts"][task["fact_id"]]
            attempt_num = task["attempt_num"]
            conversation = copy.deepcopy(task["messages"])
            conversation.append({"role": "assistant", "content": generation.get("text") or ""})
            generation_meta = {
                key: generation[key]
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "token_counter",
                    "finish_reason",
                    "stop_reason",
                    "api_model",
                    "api_base_url",
                    "prompt_token_limit",
                    "token_limit",
                    "max_tokens",
                )
                if key in generation
            }
            generation_meta["prompt_chars"] = sum(
                len(str(message.get("content") or "")) for message in task["messages"]
            )
            if result["kind"] == "validated":
                payload = {
                    "lean_code": result["lean_code"],
                    "lean_pass": bool(result["lean_pass"]),
                    "error_msg": [] if result["lean_pass"] else result["error_msg"],
                    "tries": attempt_num,
                    "conversation": conversation,
                    "generation": generation_meta,
                }
                success = bool(result["lean_pass"])
                retry_error = f"Lean error: {result['error_msg']}"
            else:
                payload = {
                    "lean_code": result.get("lean_code", ""),
                    "lean_pass": False,
                    "error_msg": result["error_msg"],
                    "tries": attempt_num,
                    "conversation": conversation,
                    "generation": generation_meta,
                }
                success = False
                retry_error = f"Error: {result['error_msg']}"

            fact["form_retries_used"] = attempt_num
            fact["formalization"] = payload
            if success:
                fact["form_status"] = "success"
            elif result["kind"] == "token_overflow" or attempt_num > self.args.formalizer_retries:
                fact["form_status"] = "failed"
            else:
                fact["form_status"] = "pending"
                fact["form_messages"] = conversation
                fact["form_messages"].append(
                    {
                        "role": "user",
                        "content": retry_error + "\n\nBased on the error, please correct the previous response. ",
                    }
                )
                await self._enqueue_form_locked(task["record_id"], task["fact_id"])
            await self._persist_record_locked(task["record_id"])

    async def _run_batch(self, worker_id: int, batch_items: List[Tuple[str, str]]) -> None:
        tasks = await self._prepare_batch(batch_items)
        if not tasks:
            return
        generations = await self._generate_outputs(worker_id, tasks)
        for task, generation in zip(tasks, generations):
            await self.validation_backpressure.acquire()
            await self.validation_queue.put((task, generation))

    def _raise_validation_error(self) -> None:
        if self.validation_error is not None:
            raise RuntimeError("FDG stage2 validation worker failed") from self.validation_error

    async def _validation_worker(self, worker_id: int) -> None:
        while True:
            task, output = await self.validation_queue.get()
            self.running_validation_items += 1
            try:
                await self._validate_and_apply_one_output(task, output)
            except Exception as exc:
                self.validation_error = exc
            finally:
                self.running_validation_items -= 1
                self.validation_queue.task_done()
                self.validation_backpressure.release()

    def _start_validation_workers(self) -> None:
        if self.validation_workers:
            return
        worker_count = max(1, self.args.lean_check_concurrency)
        self.validation_workers = [
            asyncio.create_task(self._validation_worker(worker_id))
            for worker_id in range(worker_count)
        ]

    async def _cancel_validation_workers(self) -> None:
        if not self.validation_workers:
            return
        for worker in self.validation_workers:
            worker.cancel()
        await asyncio.gather(*self.validation_workers, return_exceptions=True)
        self.validation_workers.clear()

    async def _form_worker(self, worker_id: int) -> None:
        while True:
            self._raise_validation_error()
            batch = await self._pop_batch(self._effective_form_batch_size())
            if batch:
                print(
                    f"[fdg-form:{worker_id}] backend={self.args.backend} batch={len(batch)} "
                    f"pending_validation_items={self._pending_validation_items()} "
                    f"ready_queue={self.form_queue.qsize()}"
                )
                await self._run_batch(worker_id, batch)
                continue
            async with self.state_lock:
                records_terminal = all(fdg_stage2_record_terminal(record) for record in self.records.values())
                if records_terminal and self._pending_validation_items() == 0:
                    return
            await asyncio.sleep(max(self.args.batch_wait_ms, 1) / 1000.0)

    def _write_runtime_metrics(self) -> None:
        if self.args.metrics_out is None:
            return
        total_seconds = 0.0
        if self.stage_started_perf is not None:
            total_seconds = time.perf_counter() - self.stage_started_perf
        payload = {
            "stage": "stage2",
            "graph_mode": "fdg",
            "total_execution_seconds": round(total_seconds, 6),
            "lean_compile": {
                "total_seconds": round(self.lean_compile_seconds, 6),
                "node_count": self.lean_compile_count,
                "avg_seconds_per_node": round(self.lean_compile_seconds / self.lean_compile_count, 6)
                if self.lean_compile_count
                else None,
            },
            "gpu_idle_wait": {
                "total_seconds": round(self.gpu_idle_wait_seconds, 6),
                "ratio_of_total": round(self.gpu_idle_wait_seconds / total_seconds, 6)
                if total_seconds > 0
                else None,
            },
        }
        write_json_atomic(self.args.metrics_out, payload)

    async def run(self) -> None:
        self.stage_started_perf = time.perf_counter()
        if self.args.no_resume:
            for path in (self.out_path, self.failed_path):
                if path.exists():
                    path.unlink()
            if self.checkpoint_dir.exists():
                for fp in self.checkpoint_dir.glob("*.json"):
                    fp.unlink()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)

        self.load_records()
        if not self.records:
            print("[done] no pending records.")
            return

        await self.init_runtime()
        await self.seed_ready_queue()
        self._start_validation_workers()
        try:
            worker_count = len(self.formalizers) if self.formalizers is not None else 1
            await asyncio.gather(
                *(self._form_worker(worker_id) for worker_id in range(worker_count))
            )
        except Exception as exc:
            append_jsonl(self.failed_path, {"created_at": utc_now_iso(), "error": str(exc)})
            raise
        finally:
            await self._cancel_validation_workers()
            if self.owned_lean_server and self.lean_server is not None:
                await self.lean_server.aclose()
            if self.formalizers is not None:
                close = getattr(self.formalizers, "close", None)
                if callable(close):
                    close()
                else:
                    for formalizer in self.formalizers:
                        formalizer_close = getattr(formalizer, "close", None)
                        if callable(formalizer_close):
                            formalizer_close()
            self._write_runtime_metrics()

        done = sum(1 for record in self.records.values() if fdg_stage2_record_terminal(record))
        print(f"\n[done] completed={done} out={self.out_path} failed={self.failed_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FDG Stage 2: formalize FDG facts with local vLLM or an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "graphs.jsonl",
        help="Stage 1 FDG JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage2_results.jsonl",
        help="Stage 2 final results JSONL",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage2_failed.jsonl",
        help="Stage 2 fatal failures JSONL",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage2_ckpt",
        help="Checkpoint directory for partial record states",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Max NEW records to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior outputs/checkpoints")

    parser.add_argument("--mathlib-path", default=DEFAULT_MATHLIB_PATH)
    parser.add_argument(
        "--lean-backend",
        default=DEFAULT_LEAN_BACKEND,
        choices=["subprocess", "persistent_lsp", "kimina_server"],
    )
    parser.add_argument("--lean-check-concurrency", type=int, default=16)
    parser.add_argument("--lean-worker-pool-size", type=int, default=0)
    parser.add_argument(
        "--lean-temp-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "lean_jobs",
    )
    parser.add_argument("--lean-api-url", default=DEFAULT_LEAN_API_URL)
    parser.add_argument("--lean-api-key-env", default=DEFAULT_LEAN_API_KEY_ENV)
    parser.add_argument("--lean-server-timeout", type=int, default=DEFAULT_LEAN_SERVER_TIMEOUT)
    parser.add_argument("--lean-server-reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lean-server-debug", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument(
        "--backend",
        choices=("vllm", "api"),
        default="vllm",
        help="Generation backend for Stage 2 formalization.",
    )
    parser.add_argument("--formalizer-gpus", default=DEFAULT_FORMALIZER_GPUS)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-wait-ms", type=int, default=200)
    parser.add_argument(
        "--max-pending-validation-batches",
        type=int,
        default=4,
        help="Max generated form batches allowed to wait for Lean validation.",
    )

    parser.add_argument("--formalizer-model-path", default=DEFAULT_FORMALIZER_MODEL_PATH)
    parser.add_argument("--formalizer-instances", type=int, default=1)
    parser.add_argument("--formalizer-parallel-startup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--formalizer-startup-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--formalizer-startup-timeout", type=int, default=1800)
    parser.add_argument("--formalizer-tensor-parallel-size", type=int, default=DEFAULT_FORMALIZER_TP)
    parser.add_argument("--formalizer-max-tokens", type=int, default=8192)
    parser.add_argument("--formalizer-token-limit", type=int, default=32768)
    parser.add_argument("--formalizer-temperature", type=float, default=0.0)
    parser.add_argument("--formalizer-top-p", type=float, default=1.0)
    parser.add_argument("--formalizer-presence-penalty", type=float, default=0.0)
    parser.add_argument("--formalizer-frequency-penalty", type=float, default=0.0)
    parser.add_argument("--formalizer-seed", type=int, default=42)
    parser.add_argument("--formalizer-top-k", type=int, default=20)
    parser.add_argument(
        "--formalizer-chat-template-kwargs-json",
        default=None,
        help='JSON object for tokenizer.apply_chat_template (default: {"enable_thinking": false})',
    )
    parser.add_argument(
        "--formalizer-prompt",
        default="formalize_obligation",
        help="Formalizer prompt file stem under prompts/{system,user}.",
    )
    parser.add_argument(
        "--formalizer-context-mode",
        choices=sorted(FORMALIZER_CONTEXT_MODES),
        default="c0_parent",
        help="Context supplied to the Stage 2 formalizer.",
    )
    parser.add_argument(
        "--formalizer-retries",
        type=int,
        default=3,
        help="Maximum retry rounds after the initial formalizer attempt.",
    )
    parser.add_argument("--form-batch-size", type=int, default=64)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-concurrency", type=int, default=8)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--api-retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--api-input-token-limit",
        type=int,
        default=-1,
        help="Local input-token preflight limit for API prompts (-1 uses formalizer token limit).",
    )
    parser.add_argument(
        "--api-tokenizer-path",
        default="",
        help="Optional local HF tokenizer path for API prompt token preflight.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON file for stage runtime metrics.",
    )
    return parser
