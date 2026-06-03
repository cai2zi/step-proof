from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from ..fdg_graph import (
    FDG_OUTPUT_TRUNCATED_RETRY_HINT,
    FDGDocument,
    build_fdg_messages,
    fdg_final_fact_ids,
    fdg_topo_order,
    normalize_fdg_validation_checks,
    parse_and_validate_fdg,
)
from ..fdg_stage_common import (
    FORM_TERMINAL,
    PROVE_TERMINAL,
    build_fdg_form_messages,
    build_fdg_prove_messages,
    fdg_fact_should_execute,
    fdg_stage2_checkpoint_payload,
    fdg_stage2_final_payload,
    fdg_stage2_record_terminal,
    fdg_stage3_checkpoint_payload,
    fdg_stage3_final_payload,
    fdg_stage3_record_terminal,
    fresh_fdg_stage2_record_state,
    fresh_fdg_stage3_record_state,
    restore_fdg_stage2_record_state,
    restore_fdg_stage3_record_state,
)
from ..runtime_common import (
    append_jsonl,
    extract_last_lean_block,
    load_done_ids,
    load_jsonl,
    utc_now_iso,
    write_json_atomic,
)
from .artifacts import stable_fingerprint
from .batch_engine import BatchEngine
from .context_builders import PARENT_ONLY
from .lean_runtime import LeanRuntime
from .llm_backends import LLMBackend, build_llm_backend
from .specs import LeanSpec, ModelSpec
from .stages import NodeTask, PipelineStage


JsonDict = Dict[str, Any]


def _cell_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _rel_source_file(parquet_dir: Path, fp: Path) -> str:
    try:
        return str(fp.resolve().relative_to(parquet_dir.resolve()))
    except ValueError:
        return str(fp.resolve())


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("record_id") or (payload.get("meta") or {}).get("record_id") or "").strip()


def _append_error_to_messages(
    messages: List[Dict[str, str]],
    assistant_response: str,
    error_msg: str,
) -> List[Dict[str, str]]:
    updated = list(messages)
    updated.append({"role": "assistant", "content": assistant_response})
    updated.append(
        {
            "role": "user",
            "content": (
                f"The previous response was invalid: {error_msg}\n"
                "Please correct it and return only the requested output."
            ),
        }
    )
    return updated


class GraphBuildStage(PipelineStage):
    name = "stage1"

    def __init__(self, args: argparse.Namespace, *, backend: Optional[LLMBackend] = None) -> None:
        self.args = args
        self.backend = backend
        self.stats = {"ok": 0, "skipped": 0, "failed": 0, "api_pending": 0, "retried": 0}
        self.started_perf: Optional[float] = None

    @staticmethod
    def _model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
        chat_kwargs = {"enable_thinking": False}
        if getattr(args, "chat_template_kwargs_json", None):
            chat_kwargs = json.loads(args.chat_template_kwargs_json)
        return ModelSpec(
            backend=args.backend,
            name=args.api_model if args.backend == "api" else "graph",
            model_path=args.model_path,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            instances=args.vllm_instances,
            tensor_parallel_size=args.tensor_parallel_size,
            gpus=args.gpus,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            token_limit=args.token_limit,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            seed=args.seed,
            top_k=args.top_k,
            chat_template_kwargs=chat_kwargs,
            startup_timeout=args.startup_timeout,
            parallel_startup=args.parallel_startup,
            startup_stagger_seconds=args.startup_stagger_seconds,
            api_concurrency=args.api_concurrency,
            api_timeout=args.api_timeout,
            api_max_retries=args.api_max_retries,
            api_retry_sleep=args.api_retry_sleep,
            api_input_token_limit=args.api_input_token_limit,
            api_tokenizer_path=args.api_tokenizer_path,
        )

    def _ensure_backend(self) -> LLMBackend:
        if self.backend is not None:
            return self.backend
        api_key = None
        if self.args.backend == "api":
            api_key = os.getenv(self.args.api_key_env)
            if not api_key:
                raise RuntimeError(f"--api-key-env={self.args.api_key_env!r} is not set")
        self.backend = build_llm_backend(self._model_spec_from_args(self.args), api_key=api_key)
        return self.backend

    def _done_ids(self) -> set[str]:
        done = set()
        if self.args.no_resume:
            return done
        for path in (self.args.out, self.args.failed, self.args.skipped):
            for row in load_jsonl(path):
                rid = _record_id(row)
                if rid:
                    done.add(rid)
        return done

    def _iter_records(self, done_ids: set[str]) -> Iterable[JsonDict]:
        required = [self.args.id_column, self.args.question_column, self.args.response_column]
        yielded = 0
        for fp in sorted(self.args.parquet_dir.glob(self.args.glob)):
            if not fp.is_file():
                continue
            df = pd.read_parquet(fp)
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RuntimeError(f"{fp}: missing columns {missing}; have {list(df.columns)}")
            for pos, (_, row) in enumerate(df.iterrows()):
                if self.args.limit >= 0 and yielded >= self.args.limit:
                    return
                rid = str(row[self.args.id_column]).strip()
                if not rid or rid in done_ids:
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
                yield {
                    "record_id": rid,
                    "problem": str(problem),
                    "raw_cot": str(raw_cot),
                    "source_file": _rel_source_file(self.args.parquet_dir, fp),
                    "source_row_pos": pos,
                    "messages": messages,
                    "raw_messages": copy.deepcopy(messages),
                    "retry_count": 0,
                }

    def _payload(
        self,
        record: JsonDict,
        document: FDGDocument,
        conversation: List[Dict[str, str]],
        conversation_raw: List[Dict[str, str]],
        warnings: List[JsonDict],
    ) -> JsonDict:
        return {
            "meta": {
                "schema_version": "step-proof-v2",
                "legacy_schema_version": "fdg-v1",
                "graph_mode": "fdg",
                "record_id": record["record_id"],
                "task_profile": "calc",
                "source_file": record["source_file"],
                "source_row_pos": record["source_row_pos"],
                "created_at": utc_now_iso(),
                "graph_build_tries": int(record["retry_count"]) + 1,
                "include_think_in_dag": bool(self.args.include_think_in_dag),
                "fdg_prompt": self.args.fdg_prompt,
                "validation_checks": dict(getattr(self.args, "validation_checks", {}) or {}),
                "stage_name": "stage1",
                "stage_fingerprint": self._stage_fingerprint(),
            },
            "input": {
                "problem": record["problem"],
                "raw_cot": record["raw_cot"],
            },
            "extraction": {
                "conversation": conversation,
                "conversation_raw": conversation_raw,
            },
            "graph": {
                "facts": [fact.model_dump() for fact in document.facts],
                "topo_order": fdg_topo_order(document.facts),
                "final_fact_ids": fdg_final_fact_ids(document.facts),
                "validation_warnings": warnings,
            },
        }

    def _stage_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "step-proof-v2",
                "stage": "stage1",
                "backend": self.args.backend,
                "model_path": self.args.model_path,
                "api_model": self.args.api_model,
                "fdg_prompt": self.args.fdg_prompt,
                "include_think_in_dag": self.args.include_think_in_dag,
                "validation_checks": getattr(self.args, "validation_checks", {}),
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "seed": self.args.seed,
            }
        )

    async def run(self) -> None:
        self.started_perf = time.perf_counter()
        if getattr(self.args, "validation_checks", None) is None:
            self.args.validation_checks = normalize_fdg_validation_checks(None)
        if self.args.no_resume:
            for path in (self.args.out, self.args.skipped, self.args.failed, self.args.api_pending):
                if path.exists():
                    path.unlink()
        for path in (self.args.out, self.args.skipped, self.args.failed, self.args.api_pending):
            path.parent.mkdir(parents=True, exist_ok=True)

        records: Deque[JsonDict] = deque(self._iter_records(self._done_ids()))
        print(f"[stage1:v2] pending records={len(records)}")
        if not records:
            return

        backend = self._ensure_backend()
        write_mode = "w" if self.args.no_resume else "a"
        batch_size = 1 if self.args.backend == "api" else max(1, int(self.args.batch_size))
        try:
            with (
                self.args.out.open(write_mode, encoding="utf-8") as graphs_f,
                self.args.skipped.open(write_mode, encoding="utf-8") as skipped_f,
                self.args.failed.open(write_mode, encoding="utf-8") as failed_f,
                self.args.api_pending.open(write_mode, encoding="utf-8") as api_pending_f,
            ):
                while records:
                    batch: List[JsonDict] = []
                    while records and len(batch) < batch_size:
                        batch.append(records.popleft())
                    generations = await asyncio.to_thread(
                        backend.generate_many,
                        [record["messages"] for record in batch],
                    )
                    for record, generation in zip(batch, generations):
                        self._handle_generation(
                            record,
                            generation,
                            records,
                            graphs_f,
                            skipped_f,
                            failed_f,
                            api_pending_f,
                        )
        finally:
            if self.backend is not None:
                self.backend.close()
        print(f"[stage1:v2] done stats={self.stats}")

    def _write_line(self, fp: Any, payload: JsonDict) -> None:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fp.flush()

    def _terminal_base(self, record: JsonDict) -> JsonDict:
        return {
            "record_id": record["record_id"],
            "source_file": record["source_file"],
            "source_row_pos": record["source_row_pos"],
            "fdg_prompt": self.args.fdg_prompt,
            "validation_checks": dict(getattr(self.args, "validation_checks", {}) or {}),
            "input": {
                "problem": record["problem"],
                "raw_cot": record["raw_cot"],
            },
        }

    def _handle_generation(
        self,
        record: JsonDict,
        generation: JsonDict,
        records: Deque[JsonDict],
        graphs_f: Any,
        skipped_f: Any,
        failed_f: Any,
        api_pending_f: Any,
    ) -> None:
        if generation.get("prompt_token_overflow"):
            self._write_line(
                skipped_f,
                {
                    **self._terminal_base(record),
                    "reason": "prompt_token_overflow",
                    "prompt_tokens": generation.get("prompt_tokens"),
                    "token_limit": generation.get("token_limit"),
                },
            )
            self.stats["skipped"] += 1
            return
        if generation.get("api_error"):
            self._write_line(
                api_pending_f,
                {
                    **self._terminal_base(record),
                    "schema_version": "stage1-api-pending-v1",
                    "created_at": utc_now_iso(),
                    "reason": "api_error",
                    "api_error": generation.get("api_error"),
                    "api_error_type": generation.get("api_error_type"),
                },
            )
            self.stats["api_pending"] += 1
            return

        content = generation.get("text") or ""
        conversation = list(record["messages"]) + [{"role": "assistant", "content": content}]
        conversation_raw = list(record["raw_messages"]) + [{"role": "assistant", "content": content}]
        if generation.get("output_truncated"):
            error_msg = FDG_OUTPUT_TRUNCATED_RETRY_HINT
        else:
            result = parse_and_validate_fdg(
                content,
                validation_checks=getattr(self.args, "validation_checks", None),
            )
            if result.ok and result.document is not None:
                self._write_line(
                    graphs_f,
                    self._payload(
                        record,
                        result.document,
                        conversation,
                        conversation_raw,
                        list((result.report or {}).get("warnings") or []),
                    ),
                )
                self.stats["ok"] += 1
                return
            error_msg = result.error_msg or "FDG parse/validation failed"

        if int(record["retry_count"]) < int(self.args.max_retries):
            record["retry_count"] = int(record["retry_count"]) + 1
            record["messages"] = _append_error_to_messages(record["messages"], content, error_msg)
            record["raw_messages"] = _append_error_to_messages(record["raw_messages"], content, error_msg)
            records.appendleft(record)
            self.stats["retried"] += 1
            return
        self._write_line(
            failed_f,
            {
                **self._terminal_base(record),
                "retry_count": record["retry_count"],
                "last_error": error_msg,
                "extraction": {
                    "conversation": conversation,
                    "conversation_raw": conversation_raw,
                },
            },
        )
        self.stats["failed"] += 1


class FormalizeStage(PipelineStage):
    name = "stage2"

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        backend: Optional[LLMBackend] = None,
        lean_runtime: Optional[LeanRuntime] = None,
    ) -> None:
        self.args = args
        self.records: Dict[str, JsonDict] = {}
        self.done_ids: set[str] = set()
        self.backend = backend
        self.lean_runtime = lean_runtime
        self.stage_started_perf: Optional[float] = None
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir

    @staticmethod
    def _model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
        chat_kwargs = {"enable_thinking": False}
        if getattr(args, "formalizer_chat_template_kwargs_json", None):
            chat_kwargs = json.loads(args.formalizer_chat_template_kwargs_json)
        return ModelSpec(
            backend=args.backend,
            name=args.api_model if args.backend == "api" else "formalizer",
            model_path=args.formalizer_model_path,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            instances=args.formalizer_instances,
            tensor_parallel_size=args.formalizer_tensor_parallel_size,
            gpus=args.formalizer_gpus or args.gpus,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.formalizer_max_tokens,
            token_limit=args.formalizer_token_limit,
            temperature=args.formalizer_temperature,
            top_p=args.formalizer_top_p,
            presence_penalty=args.formalizer_presence_penalty,
            frequency_penalty=args.formalizer_frequency_penalty,
            seed=args.formalizer_seed,
            top_k=args.formalizer_top_k,
            chat_template_kwargs=chat_kwargs,
            startup_timeout=args.formalizer_startup_timeout,
            parallel_startup=args.formalizer_parallel_startup,
            startup_stagger_seconds=args.formalizer_startup_stagger_seconds,
            api_concurrency=args.api_concurrency,
            api_timeout=args.api_timeout,
            api_max_retries=args.api_max_retries,
            api_retry_sleep=args.api_retry_sleep,
            api_input_token_limit=args.api_input_token_limit,
            api_tokenizer_path=args.api_tokenizer_path,
        )

    def _stage_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "step-proof-v2",
                "stage": "stage2",
                "backend": self.args.backend,
                "formalizer_prompt": self.args.formalizer_prompt,
                "formalizer_context_mode": getattr(self.args, "formalizer_context_mode", PARENT_ONLY),
                "formalizer_model_path": self.args.formalizer_model_path,
                "api_model": self.args.api_model,
                "api_base_url": self.args.api_base_url,
                "formalizer_temperature": self.args.formalizer_temperature,
                "formalizer_top_p": self.args.formalizer_top_p,
                "formalizer_seed": self.args.formalizer_seed,
            }
        )

    def _ensure_backend(self) -> LLMBackend:
        if self.backend is not None:
            return self.backend
        api_key = None
        if self.args.backend == "api":
            api_key = os.getenv(self.args.api_key_env)
            if not api_key:
                raise RuntimeError(f"--api-key-env={self.args.api_key_env!r} is not set")
        self.backend = build_llm_backend(self._model_spec_from_args(self.args), api_key=api_key)
        return self.backend

    def _ensure_lean_runtime(self) -> LeanRuntime:
        if self.lean_runtime is not None:
            return self.lean_runtime
        self.lean_runtime = LeanRuntime(
            LeanSpec(
                mathlib_path=self.args.mathlib_path,
                backend=self.args.lean_backend,
                check_concurrency=self.args.lean_check_concurrency,
                worker_pool_size=self.args.lean_worker_pool_size,
                temp_dir=self.args.lean_temp_dir,
            )
        )
        return self.lean_runtime

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
        for row in load_jsonl(self.args.infile):
            record_id = str((row.get("meta") or {}).get("record_id", "")).strip()
            if not record_id or record_id in self.done_ids:
                continue
            graph_facts = (row.get("graph") or {}).get("facts", [])
            if not graph_facts:
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
                record = restore_fdg_stage2_record_state(_load_json(ckpt_path))
            else:
                record = fresh_fdg_stage2_record_state(row)
            self._ensure_meta(record)
            self.records[record_id] = record
            if self.args.limit >= 0 and len(self.records) >= self.args.limit:
                break

    def _ensure_meta(self, record: JsonDict) -> None:
        meta = record.setdefault("meta", {})
        fingerprint = self._stage_fingerprint()
        existing = str(meta.get("stage_fingerprint") or "")
        if existing and existing != fingerprint:
            rid = meta.get("record_id", "<unknown>")
            raise RuntimeError(
                f"Record {rid} checkpoint stage_fingerprint={existing!r} "
                f"does not match current {fingerprint!r}."
            )
        meta["pipeline_schema_version"] = "step-proof-v2"
        meta["stage_name"] = "stage2"
        meta["stage_fingerprint"] = fingerprint
        meta["formalizer_prompt"] = self.args.formalizer_prompt
        meta["formalizer_context_mode"] = getattr(self.args, "formalizer_context_mode", PARENT_ONLY)

    async def _persist_record(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if fdg_stage2_record_terminal(record):
            append_jsonl(self.out_path, fdg_stage2_final_payload(record))
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            self.done_ids.add(record_id)
            return
        write_json_atomic(self.checkpoint_dir / f"{record_id}.json", fdg_stage2_checkpoint_payload(record))

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
            print("[stage2:v2] no pending records.")
            return
        await self._ensure_lean_runtime().ensure_ready()
        backend = self._ensure_backend()
        engine = BatchEngine[Tuple[str, str], NodeTask, JsonDict](
            name="stage2-formalize",
            worker_count=len(getattr(backend, "pool", [])) if self.args.backend == "vllm" else self.args.api_concurrency,
            batch_size=1 if self.args.backend == "api" else self.args.form_batch_size,
            wait_ms=self.args.batch_wait_ms,
            max_pending_validation_batches=self.args.max_pending_validation_batches,
            prepare_batch=self._prepare_batch,
            generate_batch=self._generate_batch,
            apply_generation=self._apply_generation,
            done=self._done,
        )
        for record_id, record in self.records.items():
            if fdg_stage2_record_terminal(record):
                await self._persist_record(record_id)
                continue
            for fact_id, fact in record["facts"].items():
                if fact["form_status"] == "pending":
                    fact["_form_enqueued"] = True
                    await engine.put((record_id, fact_id))
        try:
            await engine.run(validation_worker_count=self.args.lean_check_concurrency)
        finally:
            if self.backend is not None:
                self.backend.close()
            if self.lean_runtime is not None:
                await self.lean_runtime.aclose()
        print(f"[stage2:v2] done out={self.out_path} failed={self.failed_path}")

    async def _done(self) -> bool:
        return all(fdg_stage2_record_terminal(record) for record in self.records.values())

    async def _prepare_batch(self, items: List[Tuple[str, str]]) -> List[NodeTask]:
        tasks: List[NodeTask] = []
        for record_id, fact_id in items:
            record = self.records.get(record_id)
            if record is None:
                continue
            fact = record["facts"].get(fact_id)
            if fact is None:
                continue
            fact["_form_enqueued"] = False
            if not fdg_fact_should_execute(fact):
                fact["form_status"] = "skipped"
                await self._persist_record(record_id)
                continue
            if fact["form_status"] != "pending":
                continue
            fact["form_status"] = "running"
            if not fact["form_messages"]:
                fact["form_messages"] = build_fdg_form_messages(
                    fact,
                    record=record,
                    prompt_name=self.args.formalizer_prompt,
                    context_mode=getattr(self.args, "formalizer_context_mode", PARENT_ONLY),
                )
            tasks.append(
                NodeTask(
                    record_id=record_id,
                    fact_id=fact_id,
                    messages=copy.deepcopy(fact["form_messages"]),
                    attempt_num=int(fact.get("form_retries_used", 0)) + 1,
                )
            )
        return tasks

    async def _generate_batch(self, worker_id: int, tasks: List[NodeTask]) -> List[JsonDict]:
        backend = self._ensure_backend()
        return await asyncio.to_thread(backend.generate_many, [task.messages for task in tasks])

    async def _validate_generation(self, task: NodeTask, generation: JsonDict) -> JsonDict:
        if generation.get("prompt_token_overflow"):
            return {"kind": "token_overflow", "error_msg": "token_overflow", "lean_code": ""}
        if generation.get("api_error"):
            return {"kind": "api_error", "error_msg": generation.get("api_error") or "api_error", "lean_code": ""}
        if generation.get("output_truncated"):
            return {"kind": "output_truncated", "error_msg": FDG_OUTPUT_TRUNCATED_RETRY_HINT, "lean_code": ""}
        try:
            lean_code = extract_last_lean_block(generation.get("text") or "")
        except ValueError as exc:
            return {"kind": "extract_error", "error_msg": str(exc), "lean_code": ""}
        lean_pass, _lean_verify, error_msg = await self._ensure_lean_runtime().check(
            lean_code,
            job_id=f"fdg_stage2_{task.record_id}_{task.fact_id}_{task.attempt_num}",
        )
        return {"kind": "validated", "lean_code": lean_code, "lean_pass": lean_pass, "error_msg": error_msg}

    async def _apply_generation(self, task: NodeTask, generation: JsonDict) -> None:
        result = await self._validate_generation(task, generation)
        record = self.records[task.record_id]
        fact = record["facts"][task.fact_id]
        conversation = copy.deepcopy(task.messages)
        conversation.append({"role": "assistant", "content": generation.get("text") or ""})
        success = result["kind"] == "validated" and bool(result.get("lean_pass"))
        fact["form_retries_used"] = task.attempt_num
        fact["formalization"] = {
            "lean_code": result.get("lean_code", ""),
            "lean_pass": bool(success),
            "error_msg": [] if success else result.get("error_msg"),
            "tries": task.attempt_num,
            "conversation": conversation,
        }
        if success:
            fact["form_status"] = "success"
        elif result["kind"] == "token_overflow" or task.attempt_num > self.args.formalizer_retries:
            fact["form_status"] = "failed"
        else:
            fact["form_status"] = "pending"
            fact["form_messages"] = conversation + [
                {
                    "role": "user",
                    "content": f"Error: {result.get('error_msg')}\n\nBased on the error, please correct the previous response. ",
                }
            ]
        await self._persist_record(task.record_id)


class ProveStage(PipelineStage):
    name = "stage3"

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        backend: Optional[LLMBackend] = None,
        lean_runtime: Optional[LeanRuntime] = None,
    ) -> None:
        self.args = args
        self.records: Dict[str, JsonDict] = {}
        self.done_ids: set[str] = set()
        self.backend = backend
        self.lean_runtime = lean_runtime
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir

    @staticmethod
    def _model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
        chat_kwargs = {"enable_thinking": False}
        if getattr(args, "prover_chat_template_kwargs_json", None):
            chat_kwargs = json.loads(args.prover_chat_template_kwargs_json)
        return ModelSpec(
            backend="vllm",
            name="prover",
            model_path=args.prover_model_path,
            instances=args.prover_instances,
            tensor_parallel_size=args.prover_tensor_parallel_size,
            gpus=args.prover_gpus or args.gpus,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.prover_max_tokens,
            token_limit=args.prover_token_limit,
            temperature=args.prover_temperature,
            top_p=args.prover_top_p,
            presence_penalty=args.prover_presence_penalty,
            frequency_penalty=args.prover_frequency_penalty,
            seed=args.prover_seed,
            top_k=args.prover_top_k,
            chat_template_kwargs=chat_kwargs,
            startup_timeout=args.prover_startup_timeout,
            parallel_startup=args.prover_parallel_startup,
            startup_stagger_seconds=args.prover_startup_stagger_seconds,
        )

    def _stage_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "step-proof-v2",
                "stage": "stage3",
                "prover_prompt": self.args.prover_prompt,
                "prover_model_path": self.args.prover_model_path,
                "prover_temperature": self.args.prover_temperature,
                "prover_top_p": self.args.prover_top_p,
                "prover_seed": self.args.prover_seed,
            }
        )

    def _ensure_backend(self) -> LLMBackend:
        if self.backend is None:
            self.backend = build_llm_backend(self._model_spec_from_args(self.args))
        return self.backend

    def _ensure_lean_runtime(self) -> LeanRuntime:
        if self.lean_runtime is None:
            self.lean_runtime = LeanRuntime(
                LeanSpec(
                    mathlib_path=self.args.mathlib_path,
                    backend=self.args.lean_backend,
                    check_concurrency=self.args.lean_check_concurrency,
                    worker_pool_size=self.args.lean_worker_pool_size,
                    temp_dir=self.args.lean_temp_dir,
                )
            )
        return self.lean_runtime

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
        for row in load_jsonl(self.args.infile):
            record_id = str((row.get("meta") or {}).get("record_id", "")).strip()
            if not record_id or record_id in self.done_ids:
                continue
            result_facts = (row.get("results") or {}).get("facts", [])
            if not result_facts:
                append_jsonl(
                    self.failed_path,
                    {"meta": row.get("meta", {}), "error": "Empty stage2 results.", "created_at": utc_now_iso()},
                )
                continue
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if not self.args.no_resume and ckpt_path.is_file():
                record = restore_fdg_stage3_record_state(_load_json(ckpt_path))
            else:
                record = fresh_fdg_stage3_record_state(row)
            self._ensure_meta(record)
            self.records[record_id] = record
            if self.args.limit >= 0 and len(self.records) >= self.args.limit:
                break

    def _ensure_meta(self, record: JsonDict) -> None:
        meta = record.setdefault("meta", {})
        fingerprint = self._stage_fingerprint()
        existing = str(meta.get("stage_fingerprint") or "")
        if existing and existing != fingerprint:
            rid = meta.get("record_id", "<unknown>")
            raise RuntimeError(
                f"Record {rid} checkpoint stage_fingerprint={existing!r} "
                f"does not match current {fingerprint!r}."
            )
        meta["pipeline_schema_version"] = "step-proof-v2"
        meta["stage_name"] = "stage3"
        meta["stage_fingerprint"] = fingerprint
        meta["prover_prompt"] = self.args.prover_prompt

    async def _persist_record(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if fdg_stage3_record_terminal(record):
            append_jsonl(self.out_path, fdg_stage3_final_payload(record))
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            self.done_ids.add(record_id)
            return
        write_json_atomic(self.checkpoint_dir / f"{record_id}.json", fdg_stage3_checkpoint_payload(record))

    async def run(self) -> None:
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
            print("[stage3:v2] no pending records.")
            return
        await self._ensure_lean_runtime().ensure_ready()
        backend = self._ensure_backend()
        engine = BatchEngine[Tuple[str, str], NodeTask, JsonDict](
            name="stage3-prove",
            worker_count=len(getattr(backend, "pool", [])) or 1,
            batch_size=self.args.prove_batch_size,
            wait_ms=self.args.batch_wait_ms,
            max_pending_validation_batches=self.args.max_pending_validation_batches,
            prepare_batch=self._prepare_batch,
            generate_batch=self._generate_batch,
            apply_generation=self._apply_generation,
            done=self._done,
        )
        for record_id, record in self.records.items():
            if fdg_stage3_record_terminal(record):
                await self._persist_record(record_id)
                continue
            for fact_id, fact in record["facts"].items():
                if fact.get("prove_status") == "pending":
                    fact["_prove_enqueued"] = True
                    await engine.put((record_id, fact_id))
        try:
            await engine.run(validation_worker_count=self.args.lean_check_concurrency)
        finally:
            if self.backend is not None:
                self.backend.close()
            if self.lean_runtime is not None:
                await self.lean_runtime.aclose()
        print(f"[stage3:v2] done out={self.out_path} failed={self.failed_path}")

    async def _done(self) -> bool:
        return all(fdg_stage3_record_terminal(record) for record in self.records.values())

    async def _prepare_batch(self, items: List[Tuple[str, str]]) -> List[NodeTask]:
        tasks: List[NodeTask] = []
        for record_id, fact_id in items:
            record = self.records.get(record_id)
            if record is None:
                continue
            fact = record["facts"].get(fact_id)
            if fact is None:
                continue
            fact["_prove_enqueued"] = False
            if not fdg_fact_should_execute(fact):
                fact["prove_status"] = "skipped"
                await self._persist_record(record_id)
                continue
            if fact.get("prove_status") != "pending":
                continue
            fact["prove_status"] = "running"
            if not fact.get("prove_messages"):
                fact["prove_messages"] = build_fdg_prove_messages(fact, prompt_name=self.args.prover_prompt)
            if not fact.get("prove_messages_raw"):
                fact["prove_messages_raw"] = copy.deepcopy(fact["prove_messages"])
            tasks.append(
                NodeTask(
                    record_id=record_id,
                    fact_id=fact_id,
                    messages=copy.deepcopy(fact["prove_messages"]),
                    raw_messages=copy.deepcopy(fact["prove_messages_raw"]),
                    attempt_num=int(fact.get("prove_retries_used", 0)) + 1,
                )
            )
        return tasks

    async def _generate_batch(self, worker_id: int, tasks: List[NodeTask]) -> List[JsonDict]:
        return await asyncio.to_thread(self._ensure_backend().generate_many, [task.messages for task in tasks])

    async def _validate_generation(self, task: NodeTask, generation: JsonDict) -> JsonDict:
        if generation.get("prompt_token_overflow"):
            return {"kind": "token_overflow", "error_msg": "token_overflow", "lean_code": ""}
        if generation.get("output_truncated"):
            return {"kind": "output_truncated", "error_msg": FDG_OUTPUT_TRUNCATED_RETRY_HINT, "lean_code": ""}
        try:
            lean_code = extract_last_lean_block(generation.get("text") or "")
        except ValueError as exc:
            return {"kind": "extract_error", "error_msg": str(exc), "lean_code": ""}
        lean_pass, lean_verify, error_msg = await self._ensure_lean_runtime().check(
            lean_code,
            job_id=f"fdg_stage3_{task.record_id}_{task.fact_id}_{task.attempt_num}",
        )
        return {
            "kind": "validated",
            "lean_code": lean_code,
            "lean_pass": lean_pass,
            "lean_verify": lean_verify,
            "error_msg": error_msg,
        }

    async def _apply_generation(self, task: NodeTask, generation: JsonDict) -> None:
        result = await self._validate_generation(task, generation)
        record = self.records[task.record_id]
        fact = record["facts"][task.fact_id]
        conversation = copy.deepcopy(task.messages)
        conversation.append({"role": "assistant", "content": result.get("lean_code") or ""})
        conversation_raw = copy.deepcopy(task.raw_messages or task.messages)
        conversation_raw.append({"role": "assistant", "content": generation.get("text") or ""})
        success = result["kind"] == "validated" and bool(result.get("lean_verify"))
        fact["prove_retries_used"] = task.attempt_num
        fact["solved_lemma"] = {
            "lean_code": result.get("lean_code", ""),
            "lean_pass": bool(result.get("lean_pass")) if result["kind"] == "validated" else False,
            "lean_verify": bool(success),
            "error_msg": result.get("error_msg"),
            "tries": task.attempt_num,
            "conversation": conversation,
            "conversation_raw": conversation_raw,
        }
        if success:
            fact["prove_status"] = "success"
        elif result["kind"] == "token_overflow" or task.attempt_num > self.args.prover_retries:
            fact["prove_status"] = "failed"
        else:
            fact["prove_status"] = "pending"
            retry_message = {
                "role": "user",
                "content": f"Error: {result.get('error_msg')}\n\n Based on these errors, please correct the previous response. ",
            }
            fact["prove_messages"] = conversation + [retry_message]
            fact["prove_messages_raw"] = conversation_raw + [retry_message]
        await self._persist_record(task.record_id)
