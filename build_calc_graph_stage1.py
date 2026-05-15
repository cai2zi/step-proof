"""
阶段一：从 parquet 目录批量建图，写入 graph-v1 JSONL。

核心特性：
  - 本地 vLLM (vllm.LLM)：进程内加载模型，零 HTTP 开销
  - Sliding-pool batch：成功写入、失败留池、token overflow 跳过，GPU 始终满负载
  - Resume：读取已有输出 JSONL，跳过已完成 record_id
  - 独立输出：graphs.jsonl / skipped.jsonl / failed.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
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

from proofflow.fdg_graph import (
    FDGDocument,
    FDG_OUTPUT_TRUNCATED_RETRY_HINT,
    build_fdg_messages,
    fdg_origin_schema_for_prompt,
    fdg_final_fact_ids,
    fdg_topo_order,
    parse_llm_json,
    parse_and_validate_fdg,
)
from proofflow.graph_mode import FDG_GRAPH_MODE
from proofflow.llm_worker import LLMWorkerConfig, LLMWorkerPool

load_dotenv()

DEFAULT_MODEL_PATH = os.getenv("GRAPH_MODEL_PATH", "/data/czx/models/Qwen3.5-9B")
DEFAULT_TP = int(os.getenv("GRAPH_TP", "4"))
DEFAULT_GPUS = os.getenv("GRAPH_GPUS", "0,1,2,3")
DEFAULT_API_BASE_URL = os.getenv("GRAPH_API_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_MODEL = os.getenv("GRAPH_API_MODEL", "gpt-4.1")
DEFAULT_API_KEY_ENV = os.getenv("GRAPH_API_KEY_ENV", "OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendingRecord:
    record_id: str
    problem: str
    raw_cot: str
    source_file: str
    source_row_pos: int
    fdg_prompt: str
    messages: List[Dict[str, str]]
    retry_count: int = 0


@dataclass
class Stage1OutputFiles:
    graphs: Any
    skipped: Any
    failed: Any


# ---------------------------------------------------------------------------
# Helpers: parquet iteration, serialization, resume
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{rest:04.1f}s"


def _one_line(text: Any, *, limit: int = 500) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _cell_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _require_columns(df_columns: Any, required: List[str], path: Path) -> None:
    missing = [c for c in required if c not in df_columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; have {list(df_columns)}")


def _iter_parquet_rows(
    parquet_dir: Path,
    glob_pat: str,
    required_cols: List[str],
) -> Iterable[Tuple[Path, int, pd.Series]]:
    for fp in sorted(parquet_dir.glob(glob_pat)):
        if not fp.is_file():
            continue
        df = pd.read_parquet(fp)
        _require_columns(df.columns, required_cols, fp)
        for pos, (_, row) in enumerate(df.iterrows()):
            yield fp, pos, row


def _rel_source_file(parquet_dir: Path, fp: Path) -> str:
    try:
        return str(fp.resolve().relative_to(parquet_dir.resolve()))
    except ValueError:
        return str(fp.resolve())


def load_done_ids(out_path: Path, *, expected_fdg_prompt: str | None = None) -> set:
    """Read already-completed record_ids from an existing output JSONL."""
    done: set = set()
    if not out_path.is_file():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                meta = obj.get("meta", {})
                prompt = meta.get("fdg_prompt")
                if expected_fdg_prompt is not None and prompt != expected_fdg_prompt:
                    rid = meta.get("record_id", "<unknown>")
                    raise SystemExit(
                        f"{out_path}: record {rid} was created with fdg_prompt={prompt!r}, "
                        f"but current config uses {expected_fdg_prompt!r}. "
                        "Use a new exp.name, pass --no-resume, or remove the old output file."
                    )
                rid = meta.get("record_id")
                if rid:
                    done.add(str(rid))
            except json.JSONDecodeError:
                pass
    return done


def load_terminal_ids(path: Path) -> set:
    """Read record_ids from terminal side outputs such as skipped.jsonl/failed.jsonl."""
    terminal: set = set()
    if not path.is_file():
        return terminal
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("record_id") or (obj.get("meta") or {}).get("record_id")
            if rid:
                terminal.add(str(rid))
    return terminal


class Stage1Progress:
    """Total record-level progress for Stage 1 terminal outcomes."""

    def __init__(self, total: int, *, started: Optional[float] = None) -> None:
        self.total = max(0, int(total))
        self.completed = 0
        self.started = time.perf_counter() if started is None else started
        self._bar = None
        try:
            from tqdm.auto import tqdm  # type: ignore

            self._bar = tqdm(
                total=self.total,
                desc="[stage1]",
                unit="record",
                dynamic_ncols=True,
            )
        except Exception:
            self._bar = None
            print(self._line({}), flush=True)

    def _snapshot(self, stats: Dict[str, int]) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        remaining = max(0, self.total - self.completed)
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        eta = remaining / rate if rate > 0 else 0.0
        return {
            "completed": self.completed,
            "total": self.total,
            "remaining": remaining,
            "elapsed": elapsed,
            "eta": eta,
            "rate": rate,
            "ok": stats.get("ok", 0),
            "skipped": stats.get("skipped", 0),
            "failed": stats.get("failed", 0),
            "retried": stats.get("retried", 0),
        }

    def _postfix(self, stats: Dict[str, int]) -> str:
        snap = self._snapshot(stats)
        return (
            f"completed={snap['completed']}/{snap['total']} "
            f"remaining={snap['remaining']} "
            f"elapsed={_fmt_seconds(snap['elapsed'])} "
            f"eta={_fmt_seconds(snap['eta']) if snap['completed'] else 'unknown'} "
            f"ok={snap['ok']} skip={snap['skipped']} "
            f"fail={snap['failed']} retry={snap['retried']}"
        )

    def _line(self, stats: Dict[str, int]) -> str:
        return f"[progress] {self._postfix(stats)}"

    def log(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)
        else:
            print(message, flush=True)

    def update(self, amount: int, stats: Dict[str, int]) -> None:
        if amount <= 0:
            return
        self.completed += amount
        if self._bar is not None:
            self._bar.set_postfix_str(self._postfix(stats), refresh=False)
            self._bar.update(amount)
        else:
            print(self._line(stats), flush=True)

    def close(self, stats: Dict[str, int]) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(self._postfix(stats), refresh=False)
            self._bar.close()


class PendingRecordSource:
    """Scans parquet shards and yields Stage 1 records after resume filtering."""

    def __init__(self, args: argparse.Namespace, done_ids: set, log: Any = print) -> None:
        self.args = args
        self.done_ids = done_ids
        self.log = log
        self.required_columns = [
            args.id_column,
            args.question_column,
            args.response_column,
        ]

    def _rows(self) -> Iterable[Tuple[Path, int, pd.Series]]:
        return _iter_parquet_rows(self.args.parquet_dir, self.args.glob, self.required_columns)

    def _candidate_values(self, row: pd.Series) -> Optional[Tuple[str, Any, Any]]:
        record_id = str(row[self.args.id_column]).strip()
        if not record_id or record_id in self.done_ids:
            return None
        problem = row[self.args.question_column]
        raw_cot = row[self.args.response_column]
        if _cell_missing(problem) or _cell_missing(raw_cot):
            return None
        return record_id, problem, raw_cot

    def count_pending(self) -> int:
        total = 0
        for _, _, row in self._rows():
            if self.args.limit >= 0 and total >= self.args.limit:
                break
            if self._candidate_values(row) is not None:
                total += 1
        return total

    def iter_pending(self) -> Iterable[PendingRecord]:
        yielded = 0
        for fp, pos, row in self._rows():
            if self.args.limit >= 0 and yielded >= self.args.limit:
                return
            record_id = str(row[self.args.id_column]).strip()
            if not record_id:
                continue
            if record_id in self.done_ids:
                self.log(f"  [resume] skip {record_id}")
                continue
            problem = row[self.args.question_column]
            raw_cot = row[self.args.response_column]
            if _cell_missing(problem) or _cell_missing(raw_cot):
                continue
            messages = build_fdg_messages(
                problem_text=str(problem),
                solution_or_cot=str(raw_cot),
                include_think_in_dag=self.args.include_think_in_dag,
                prompt_name=self.args.fdg_prompt,
            )
            yielded += 1
            yield PendingRecord(
                record_id=record_id,
                problem=str(problem),
                raw_cot=str(raw_cot),
                source_file=_rel_source_file(self.args.parquet_dir, fp),
                source_row_pos=pos,
                fdg_prompt=self.args.fdg_prompt,
                messages=messages,
            )


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _build_fdg_payload(
    record: PendingRecord,
    document: FDGDocument,
    tries: int,
    include_think_in_dag: bool,
    conversation: List[Dict[str, str]],
    validation_warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "schema_version": "fdg-v1",
            "graph_mode": FDG_GRAPH_MODE,
            "record_id": record.record_id,
            "task_profile": "calc",
            "source_file": record.source_file,
            "source_row_pos": record.source_row_pos,
            "created_at": _utc_now_iso(),
            "graph_build_tries": tries,
            "include_think_in_dag": include_think_in_dag,
            "fdg_prompt": record.fdg_prompt,
            "origin_schema": fdg_origin_schema_for_prompt(record.fdg_prompt),
        },
        "input": {
            "problem": record.problem,
            "raw_cot": record.raw_cot,
        },
        "extraction": {
            "conversation": conversation,
        },
        "graph": {
            "facts": [fact.model_dump() for fact in document.facts],
            "topo_order": fdg_topo_order(document.facts),
            "final_fact_ids": fdg_final_fact_ids(document.facts),
            "validation_warnings": validation_warnings,
        },
    }


def compact_fdg_response_for_retry(assistant_response: str) -> str:
    """Keep only the extracted FDG payload in retry context when possible."""
    if not assistant_response:
        return ""
    try:
        parsed = parse_llm_json(assistant_response)
    except Exception:
        return ""
    try:
        parsed = FDGDocument.model_validate(parsed).model_dump()
    except Exception:
        pass
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def append_error_to_messages(
    messages: List[Dict[str, str]],
    assistant_response: str,
    error_msg: str,
) -> List[Dict[str, str]]:
    updated = list(messages)
    updated.append(
        {
            "role": "assistant",
            "content": compact_fdg_response_for_retry(assistant_response),
        }
    )
    updated.append(
        {
            "role": "user",
            "content": f"The previous FDG JSON was invalid: {error_msg}\nPlease correct it and output only valid FDG JSON.",
        }
    )
    return updated


# ---------------------------------------------------------------------------
# API backend
# ---------------------------------------------------------------------------

class Stage1APIClient:
    """OpenAI-compatible chat API backend for Stage 1 graph generation."""

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
        # Local preflight only; provider billing/context accounting can still differ.
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

            last_error = ""
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
                    last_error = str(exc)
                    error_history.append(
                        {
                            "attempt": attempts,
                            "error_type": type(exc).__name__,
                            "error": last_error,
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
                                "api_error": last_error,
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

    def generate_one_with_metadata(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        return self.generate_with_metadata([messages])[0]

    def close(self, force: bool = False) -> None:
        client = getattr(self._local, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


class Stage1ResultHandler:
    def __init__(
        self,
        args: argparse.Namespace,
        pool: deque[PendingRecord],
        stats: Dict[str, int],
        timing_stats: Dict[str, float],
        progress: Stage1Progress,
        log: Any = print,
    ) -> None:
        self.args = args
        self.pool = pool
        self.stats = stats
        self.timing_stats = timing_stats
        self.progress = progress
        self.log = log

    def _write_jsonl(self, fp: Any, payload: Dict[str, Any]) -> None:
        write_started = time.perf_counter()
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fp.flush()
        self.timing_stats["write_seconds"] += time.perf_counter() - write_started

    def _record_input(self, record: PendingRecord) -> Dict[str, str]:
        return {
            "problem": record.problem,
            "raw_cot": record.raw_cot,
        }

    def _terminal_base_payload(self, record: PendingRecord) -> Dict[str, Any]:
        return {
            "record_id": record.record_id,
            "source_file": record.source_file,
            "source_row_pos": record.source_row_pos,
            "fdg_prompt": record.fdg_prompt,
            "origin_schema": fdg_origin_schema_for_prompt(record.fdg_prompt),
        }

    def _mark_terminal(self, key: str) -> None:
        self.stats[key] += 1
        self.progress.update(1, self.stats)

    def _record_retry(self, record: PendingRecord, assistant_response: str, error_msg: str) -> None:
        retry_started = time.perf_counter()
        record.messages = append_error_to_messages(record.messages, assistant_response, error_msg)
        record.retry_count += 1
        self.pool.appendleft(record)
        self.timing_stats["retry_prepare_seconds"] += time.perf_counter() - retry_started
        self.stats["retried"] += 1

    def _write_failed(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        error_msg: str,
        conversation: List[Dict[str, str]],
        generation: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            **self._terminal_base_payload(record),
            "retry_count": record.retry_count,
            "last_error": error_msg,
            "input": self._record_input(record),
            "extraction": {
                "conversation": conversation,
            },
        }
        if generation is not None:
            payload["finish_reason"] = generation.get("finish_reason")
            payload["stop_reason"] = generation.get("stop_reason")
        self._write_jsonl(outputs.failed, payload)
        self._mark_terminal("failed")

    def _log_api_timing(self, record: PendingRecord, generation: Dict[str, Any]) -> None:
        api_error = generation.get("api_error")
        self.log(
            f"  [api-timing] {record.record_id} "
            f"total={_fmt_seconds(float(generation.get('api_total_seconds') or 0.0))} "
            f"request={_fmt_seconds(float(generation.get('api_request_seconds') or 0.0))} "
            f"tokenize={_fmt_seconds(float(generation.get('token_count_seconds') or 0.0))} "
            f"sleep={_fmt_seconds(float(generation.get('api_sleep_seconds') or 0.0))} "
            f"attempts={generation.get('api_attempts')} "
            f"prompt_tokens={generation.get('prompt_tokens')} "
            f"completion_tokens={generation.get('completion_tokens')} "
            f"chars={generation.get('response_chars')}"
            + (
                f" error_type={generation.get('api_error_type')} "
                f"error={_one_line(api_error)}"
                if api_error
                else ""
            )
        )
        for err in generation.get("api_error_history") or []:
            self.log(
                f"    [api-error] attempt={err.get('attempt')} "
                f"elapsed={_fmt_seconds(float(err.get('elapsed_seconds') or 0.0))} "
                f"type={err.get('error_type')} "
                f"message={_one_line(err.get('error'))}"
            )

    def _accumulate_generation_timing(self, generation: Dict[str, Any]) -> None:
        self.timing_stats["api_token_count_seconds"] += float(generation.get("token_count_seconds") or 0.0)
        self.timing_stats["api_request_seconds"] += float(generation.get("api_request_seconds") or 0.0)
        self.timing_stats["api_sleep_seconds"] += float(generation.get("api_sleep_seconds") or 0.0)
        self.timing_stats["api_total_seconds"] += float(generation.get("api_total_seconds") or 0.0)

    def _handle_prompt_overflow(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        generation: Dict[str, Any],
    ) -> None:
        skipped_token_limit = generation.get(
            "token_limit",
            self.args.api_input_token_limit if self.args.backend == "api" else self.args.token_limit,
        )
        self._write_jsonl(
            outputs.skipped,
            {
                **self._terminal_base_payload(record),
                "reason": "prompt_token_overflow",
                "prompt_tokens": generation.get("prompt_tokens"),
                "prompt_token_limit": generation.get("prompt_token_limit"),
                "token_limit": skipped_token_limit,
                "max_tokens": generation.get("max_tokens"),
                "token_counter": generation.get("token_counter"),
                "input": self._record_input(record),
                "extraction": {
                    "conversation": list(record.messages),
                },
            },
        )
        self._mark_terminal("skipped")
        self.log(f"  [skip]  {record.record_id} (prompt token overflow)")

    def _handle_api_error(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        generation: Dict[str, Any],
    ) -> None:
        error_msg = f"API request failed: {generation.get('api_error')}"
        conversation = list(record.messages) + [{"role": "assistant", "content": ""}]
        if record.retry_count >= self.args.max_retries:
            self._write_failed(outputs, record, error_msg, conversation, generation)
            self.log(
                f"  [fail]  {record.record_id}  "
                f"(API error after {record.retry_count} retries: "
                f"{generation.get('api_error_type')}: {_one_line(generation.get('api_error'))})"
            )
            return

        self._record_retry(record, "", error_msg)
        self.log(
            f"  [retry] {record.record_id}  "
            f"(API error; attempt {record.retry_count}/{self.args.max_retries}; "
            f"{generation.get('api_error_type')}: {_one_line(generation.get('api_error'))})"
        )

    def _handle_truncated_output(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        content: str,
        conversation: List[Dict[str, str]],
        generation: Dict[str, Any],
    ) -> None:
        error_msg = FDG_OUTPUT_TRUNCATED_RETRY_HINT
        if record.retry_count >= self.args.max_retries:
            self._write_failed(outputs, record, error_msg, conversation, generation)
            self.log(f"  [fail]  {record.record_id}  (output truncated after {record.retry_count} retries)")
            return

        self._record_retry(record, content, error_msg)
        self.log(f"  [retry] {record.record_id}  (output truncated; attempt {record.retry_count}/{self.args.max_retries})")

    def _handle_valid_fdg(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        document: FDGDocument,
        conversation: List[Dict[str, str]],
        warnings: List[Dict[str, Any]],
        parse_seconds: float,
    ) -> None:
        payload_started = time.perf_counter()
        payload = _build_fdg_payload(
            record,
            document,
            record.retry_count + 1,
            self.args.include_think_in_dag,
            conversation,
            warnings,
        )
        payload_seconds = time.perf_counter() - payload_started
        self.timing_stats["payload_build_seconds"] += payload_seconds

        write_started = time.perf_counter()
        outputs.graphs.write(json.dumps(payload, ensure_ascii=False) + "\n")
        outputs.graphs.flush()
        write_seconds = time.perf_counter() - write_started
        self.timing_stats["write_seconds"] += write_seconds
        self._mark_terminal("ok")
        self.log(
            f"  [ok]    {record.record_id}  "
            f"(tries={record.retry_count + 1} "
            f"parse={_fmt_seconds(parse_seconds)} "
            f"payload={_fmt_seconds(payload_seconds)} "
            f"write={_fmt_seconds(write_seconds)})"
        )

    def _handle_invalid_fdg(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        content: str,
        conversation: List[Dict[str, str]],
        error_msg: str,
    ) -> None:
        if record.retry_count >= self.args.max_retries:
            self._write_failed(outputs, record, error_msg, conversation)
            self.log(f"  [fail]  {record.record_id}  (after {record.retry_count} retries)")
            return

        self._record_retry(record, content, error_msg)
        self.log(f"  [retry] {record.record_id}  (attempt {record.retry_count}/{self.args.max_retries})")

    def handle(
        self,
        outputs: Stage1OutputFiles,
        record: PendingRecord,
        generation: Dict[str, Any],
    ) -> None:
        self._accumulate_generation_timing(generation)
        if self.args.backend == "api":
            self._log_api_timing(record, generation)

        if generation.get("prompt_token_overflow"):
            self._handle_prompt_overflow(outputs, record, generation)
            return

        if generation.get("api_error"):
            self._handle_api_error(outputs, record, generation)
            return

        content = generation.get("text")
        if content is None:
            content = ""
        conversation = list(record.messages) + [{"role": "assistant", "content": content}]

        if generation.get("output_truncated"):
            self._handle_truncated_output(outputs, record, content, conversation, generation)
            return

        parse_started = time.perf_counter()
        result = parse_and_validate_fdg(content, prompt_name=record.fdg_prompt)
        parse_seconds = time.perf_counter() - parse_started
        self.timing_stats["parse_validate_seconds"] += parse_seconds
        if result.ok:
            self._handle_valid_fdg(
                outputs,
                record,
                result.document,
                conversation,
                list((result.report or {}).get("warnings") or []),
                parse_seconds,
            )
            return

        self._handle_invalid_fdg(outputs, record, content, conversation, result.error_msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 (batch + local vLLM): build calc proof graphs → graph-v1 JSONL.",
    )
    # Input
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="Directory containing .parquet shards")
    parser.add_argument("--glob", default="*.parquet",
                        help="Glob pattern under parquet-dir (default: *.parquet)")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max NEW records to process (-1 = all)")
    # Output
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "graphs.jsonl",
                        help="Successful graphs output JSONL")
    parser.add_argument("--skipped", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "skipped.jsonl",
                        help="Token-overflow skipped records JSONL")
    parser.add_argument("--failed", type=Path,
                        default=Path(__file__).resolve().parent / "calc_runs" / "failed.jsonl",
                        help="Permanently failed records JSONL")
    # Resume
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing output and start fresh (overwrite)")
    # Backend
    parser.add_argument("--backend", choices=("vllm", "api"), default="vllm",
                        help="Generation backend for Stage 1 graph building")
    # vLLM
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vllm-instances", type=int, default=1)
    parser.add_argument("--parallel-startup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--startup-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP)
    parser.add_argument("--gpus", default=DEFAULT_GPUS,
                        help="CUDA_VISIBLE_DEVICES for the local vLLM engine")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help='JSON object for tokenizer.apply_chat_template (default: {"enable_thinking": false})',
    )
    parser.add_argument("--token-limit", type=int, default=40960,
                        help="Max prompt tokens; longer prompts are skipped")
    # OpenAI-compatible API
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL,
                        help="OpenAI-compatible chat model name for --backend api")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL,
                        help="OpenAI-compatible base URL, e.g. OpenAI or Qwen compatible mode")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                        help="Environment variable containing the API key")
    parser.add_argument("--api-concurrency", type=int, default=8,
                        help="Number of concurrent API request workers")
    parser.add_argument("--api-timeout", type=float, default=120.0,
                        help="Per-request API timeout in seconds")
    parser.add_argument("--api-max-retries", type=int, default=3,
                        help="Retries for transient API request failures")
    parser.add_argument("--api-retry-sleep", type=float, default=2.0,
                        help="Base sleep seconds between API request retries")
    parser.add_argument("--api-input-token-limit", type=int, default=-1,
                        help="Local input-token preflight limit for API prompts (-1 disables)")
    parser.add_argument("--api-tokenizer-path", default="",
                        help="Optional local HF tokenizer path for API prompt token preflight")
    # Batch / retry
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    # FDG options
    parser.add_argument(
        "--fdg-prompt",
        default="fdg",
        help="FDG prompt file stem under prompts/{system,user}.",
    )
    parser.add_argument(
        "--include-think-in-dag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow FDG extraction to see <think>...</think> content in the response. "
            "Use --no-include-think-in-dag to hide think blocks from the graph model; "
            "the full raw_cot is still written to output JSONL."
        ),
    )

    return parser


def main() -> None:
    run_started = time.perf_counter()
    args = build_arg_parser().parse_args()

    chat_template_kwargs: Optional[Dict[str, Any]] = None
    if args.chat_template_kwargs_json:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit("--chat-template-kwargs-json must be a JSON object")

    if not args.parquet_dir.is_dir():
        raise SystemExit(f"--parquet-dir is not a directory: {args.parquet_dir}")

    setup_started = time.perf_counter()

    # Resume: load already-done IDs
    done_ids: set = set()
    if not args.no_resume:
        done_ids = load_done_ids(args.out, expected_fdg_prompt=args.fdg_prompt)
        failed_ids = load_terminal_ids(args.failed)
        skipped_ids = load_terminal_ids(args.skipped)
        done_ids.update(failed_ids)
        done_ids.update(skipped_ids)
        if done_ids:
            print(f"[resume] skipping {len(done_ids)} already-processed record(s)")
        if failed_ids:
            print(f"[resume] includes {len(failed_ids)} previously failed record(s)")
        if skipped_ids:
            print(f"[resume] includes {len(skipped_ids)} previously skipped record(s)")

    # Prepare output dirs
    for p in (args.out, args.skipped, args.failed):
        p.parent.mkdir(parents=True, exist_ok=True)

    source = PendingRecordSource(args, done_ids)
    total_pending = source.count_pending()
    print(f"[progress] total pending records to process: {total_pending}")
    if total_pending <= 0:
        print("[done] no pending records.")
        return

    progress = Stage1Progress(total_pending, started=run_started)
    pending_iter = iter(PendingRecordSource(args, done_ids, log=progress.log).iter_pending())

    pool: deque[PendingRecord] = deque()
    worker_count = args.vllm_instances if args.backend == "vllm" else args.api_concurrency
    if worker_count < 1:
        raise SystemExit("--vllm-instances/--api-concurrency must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    effective_batch_size = args.batch_size if args.backend == "vllm" else 1
    refill_target = worker_count * effective_batch_size

    def refill_pool() -> None:
        while len(pool) < refill_target:
            rec = next(pending_iter, None)
            if rec is None:
                break
            pool.append(rec)

    refill_pool()
    if not pool:
        print("[done] no pending records.")
        progress.close({"ok": 0, "skipped": 0, "failed": 0, "retried": 0})
        return

    stats = {"ok": 0, "skipped": 0, "failed": 0, "retried": 0}
    timing_stats = {
        "setup_seconds": time.perf_counter() - setup_started,
        "dispatch_seconds": 0.0,
        "generation_wait_seconds": 0.0,
        "api_token_count_seconds": 0.0,
        "api_request_seconds": 0.0,
        "api_sleep_seconds": 0.0,
        "api_total_seconds": 0.0,
        "parse_validate_seconds": 0.0,
        "payload_build_seconds": 0.0,
        "write_seconds": 0.0,
        "retry_prepare_seconds": 0.0,
    }

    llm_pool: Any
    if args.backend == "vllm":
        # Load local vLLM workers after discovering pending work.
        print(
            f"[init] loading model {args.model_path} "
            f"(instances={args.vllm_instances}, tp={args.tensor_parallel_size}, gpus={args.gpus}) ..."
        )
        llm_pool = LLMWorkerPool(
            base_config=LLMWorkerConfig(
                name="graph",
                gpus="",
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                seed=args.seed,
                top_k=args.top_k,
                token_limit=args.token_limit,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                chat_template_kwargs=(
                    chat_template_kwargs
                    if chat_template_kwargs is not None
                    else {"enable_thinking": False}
                ),
            ),
            instances=args.vllm_instances,
            gpus=args.gpus,
            startup_timeout=args.startup_timeout,
            parallel_startup=args.parallel_startup,
            startup_stagger_seconds=args.startup_stagger_seconds,
        )
        print(f"[init] model workers ready: {llm_pool.gpu_groups}\n")
    else:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"--api-key-env={args.api_key_env!r} is not set")
        input_token_limit = args.api_input_token_limit
        if input_token_limit < 0:
            input_token_limit = args.token_limit
        print(
            f"[init] using API backend model={args.api_model} "
            f"base_url={args.api_base_url} concurrency={args.api_concurrency} "
            f"input_token_limit={input_token_limit} "
            f"api_tokenizer_path={args.api_tokenizer_path or '<tiktoken>'} "
            f"effective_batch_size={effective_batch_size}"
        )
        llm_pool = [
            Stage1APIClient(
                model=args.api_model,
                base_url=args.api_base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                input_token_limit=input_token_limit,
                tokenizer_path=args.api_tokenizer_path,
                tokenizer_chat_template_kwargs=chat_template_kwargs,
                timeout=args.api_timeout,
                api_max_retries=args.api_max_retries,
                retry_sleep=args.api_retry_sleep,
            )
            for _ in range(worker_count)
        ]
        print("[init] API workers ready\n")

    write_mode = "w" if args.no_resume else "a"

    def next_batch() -> List[PendingRecord]:
        batch: List[PendingRecord] = []
        while len(batch) < effective_batch_size and pool:
            batch.append(pool.popleft())
        return batch

    def dispatch_next(
        executor: concurrent.futures.ThreadPoolExecutor,
        future_to_batch: Dict[concurrent.futures.Future, Tuple[int, List[PendingRecord], float]],
        worker_id: int,
    ) -> bool:
        dispatch_started = time.perf_counter()
        refill_pool()
        batch = next_batch()
        if not batch:
            timing_stats["dispatch_seconds"] += time.perf_counter() - dispatch_started
            return False
        if args.backend == "api":
            record = batch[0]
            future = executor.submit(
                llm_pool[worker_id].generate_one_with_metadata,
                record.messages,
            )
        else:
            future = executor.submit(
                llm_pool[worker_id].generate_with_metadata,
                [record.messages for record in batch],
            )
        dispatched_at = time.perf_counter()
        future_to_batch[future] = (worker_id, batch, dispatched_at)
        timing_stats["dispatch_seconds"] += dispatched_at - dispatch_started
        if args.backend == "api":
            progress.log(f"  [worker:{worker_id}] dispatched record={batch[0].record_id} retry={batch[0].retry_count}")
        else:
            progress.log(f"  [worker:{worker_id}] dispatched batch size={len(batch)}")
        return True

    try:
        with (
            open(args.out, write_mode, encoding="utf-8") as graphs_f,
            open(args.skipped, write_mode, encoding="utf-8") as skipped_f,
            open(args.failed, write_mode, encoding="utf-8") as failed_f,
            concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor,
        ):
            outputs = Stage1OutputFiles(graphs=graphs_f, skipped=skipped_f, failed=failed_f)
            result_handler = Stage1ResultHandler(
                args,
                pool,
                stats,
                timing_stats,
                progress,
                log=progress.log,
            )
            future_to_batch: Dict[concurrent.futures.Future, Tuple[int, List[PendingRecord], float]] = {}
            for worker_id in range(worker_count):
                dispatch_next(executor, future_to_batch, worker_id)

            completed_batches = 0
            while future_to_batch:
                wait_started = time.perf_counter()
                done, _ = concurrent.futures.wait(
                    future_to_batch,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                timing_stats["generation_wait_seconds"] += time.perf_counter() - wait_started
                for future in done:
                    worker_id, batch, dispatched_at = future_to_batch[future]
                    del future_to_batch[future]
                    future_result = future.result()
                    generations = [future_result] if args.backend == "api" else future_result
                    batch_wall_seconds = time.perf_counter() - dispatched_at
                    completed_batches += 1
                    progress.log(
                        f"\n[worker:{worker_id}] completed batch size={len(batch)} "
                        f"completed_batches={completed_batches} "
                        f"wall={_fmt_seconds(batch_wall_seconds)} "
                        f"(ok={stats['ok']} skip={stats['skipped']} fail={stats['failed']})"
                    )

                    for record, generation in zip(batch, generations):
                        result_handler.handle(outputs, record, generation)

                    dispatch_next(executor, future_to_batch, worker_id)
    finally:
        if hasattr(llm_pool, "close"):
            llm_pool.close()
        else:
            for client in llm_pool:
                client.close()
        progress.close(stats)

    print(
        f"\n[done] ok={stats['ok']}  skipped={stats['skipped']}  "
        f"failed={stats['failed']}  retried={stats['retried']}"
    )
    print(f"  graphs  → {args.out}")
    print(f"  skipped → {args.skipped}")
    print(f"  failed  → {args.failed}")

    total_seconds = time.perf_counter() - run_started
    processed = stats["ok"] + stats["skipped"] + stats["failed"]
    rate = processed / total_seconds if total_seconds > 0 else 0.0
    print(
        "[timing] "
        f"total={_fmt_seconds(total_seconds)} "
        f"processed={processed} rate={rate:.3f}/s "
        f"setup={_fmt_seconds(timing_stats['setup_seconds'])} "
        f"dispatch={_fmt_seconds(timing_stats['dispatch_seconds'])} "
        f"wait={_fmt_seconds(timing_stats['generation_wait_seconds'])} "
        f"api_total_sum={_fmt_seconds(timing_stats['api_total_seconds'])} "
        f"api_request_sum={_fmt_seconds(timing_stats['api_request_seconds'])} "
        f"api_tokenize_sum={_fmt_seconds(timing_stats['api_token_count_seconds'])} "
        f"api_sleep_sum={_fmt_seconds(timing_stats['api_sleep_seconds'])} "
        f"parse_sum={_fmt_seconds(timing_stats['parse_validate_seconds'])} "
        f"payload_sum={_fmt_seconds(timing_stats['payload_build_seconds'])} "
        f"write_sum={_fmt_seconds(timing_stats['write_seconds'])} "
        f"retry_prepare_sum={_fmt_seconds(timing_stats['retry_prepare_seconds'])}"
    )


if __name__ == "__main__":
    main()
