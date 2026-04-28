from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .fdg_stage_common import (
    FORM_TERMINAL,
    build_fdg_form_messages,
    fdg_stage2_checkpoint_payload,
    fdg_stage2_final_payload,
    fdg_stage2_record_terminal,
    fresh_fdg_stage2_record_state,
    restore_fdg_stage2_record_state,
)
from .lean_check import LeanServer
from .llm_worker import LLMWorkerClient, LLMWorkerConfig
from .stage2_common import (
    append_jsonl,
    extract_last_lean_block,
    load_done_ids,
    load_jsonl,
    utc_now_iso,
    write_json_atomic,
)

load_dotenv()

DEFAULT_FORMALIZER_MODEL_PATH = os.getenv(
    "FORMALIZER_MODEL_PATH", "/data/czx/models/Goedel-Formalizer-V2-8B"
)
DEFAULT_GPUS = os.getenv("STAGE2_GPUS", os.getenv("GRAPH_GPUS", "0,1,2,3"))
DEFAULT_MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")


class FDGStage2Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.records: Dict[str, Dict[str, Any]] = {}
        self.form_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.state_lock = asyncio.Lock()
        self.lean_semaphore = asyncio.Semaphore(args.lean_check_concurrency)
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir
        self.done_ids: set[str] = set()
        self.formalizer: Optional[LLMWorkerClient] = None
        self.lean_server: Optional[LeanServer] = None
        self.validation_backpressure_limit = max(1, args.max_pending_validation_batches) * max(1, args.form_batch_size)
        self.validation_backpressure = asyncio.Semaphore(self.validation_backpressure_limit)
        self.validation_queue: asyncio.Queue[Tuple[Dict[str, Any], Optional[str]]] = asyncio.Queue()
        self.validation_workers: List[asyncio.Task] = []
        self.running_validation_items = 0
        self.validation_error: Optional[BaseException] = None
        self.stage_started_perf: Optional[float] = None
        self.gpu_idle_wait_seconds = 0.0
        self.lean_compile_seconds = 0.0
        self.lean_compile_count = 0

    def _pending_validation_items(self) -> int:
        return self.validation_queue.qsize() + self.running_validation_items

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
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
                partial_count += 1
                for fact in record["facts"].values():
                    if fact["form_status"] in FORM_TERMINAL:
                        form_done += 1
            else:
                record = fresh_fdg_stage2_record_state(row)

            self.records[record_id] = record
            loaded += 1
            if self.args.limit >= 0 and loaded >= self.args.limit:
                break

        print(f"\n[resume] Fully completed records skipped: {resumed_count}")
        print(f"[resume] Empty FDG records skipped: {empty_skipped}")
        print(f"[resume] Partially completed records loaded: {partial_count}")
        print(f"[resume] Facts already formalized/skipped: {form_done}")
        print(f"[resume] Total pending records to process: {loaded}\n")

    def _resolve_formalizer_gpus(self) -> str:
        if self.args.formalizer_gpus:
            return self.args.formalizer_gpus
        all_devices = [device.strip() for device in self.args.gpus.split(",") if device.strip()]
        total_needed = self.args.formalizer_tensor_parallel_size
        if len(all_devices) < total_needed:
            raise RuntimeError(
                "Not enough GPUs in --gpus to derive the formalizer worker. "
                f"Need {total_needed}, got {len(all_devices)} from {self.args.gpus!r}."
            )
        return ",".join(all_devices[:total_needed])

    async def init_runtime(self) -> None:
        formalizer_gpus = self._resolve_formalizer_gpus()
        print(
            "[init] loading formalizer",
            self.args.formalizer_model_path,
            f"(tp={self.args.formalizer_tensor_parallel_size}, gpus={formalizer_gpus}) ...",
        )
        chat_template_kwargs: Dict[str, Any] = {"enable_thinking": False}
        if self.args.formalizer_chat_template_kwargs_json:
            parsed = json.loads(self.args.formalizer_chat_template_kwargs_json)
            if not isinstance(parsed, dict):
                raise RuntimeError("--formalizer-chat-template-kwargs-json must be a JSON object")
            chat_template_kwargs = parsed

        self.formalizer = LLMWorkerClient(
            config=LLMWorkerConfig(
                name="formalizer",
                gpus=formalizer_gpus,
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
            )
        )
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

    async def _validate_batch_output(self, task: Dict[str, Any], output: Optional[str]) -> Dict[str, Any]:
        if output is None:
            return {"kind": "token_overflow", "error_msg": "token_overflow", "lean_code": ""}
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
                if fact["form_status"] != "pending":
                    continue
                fact["form_status"] = "running"
                if not fact["form_messages"]:
                    fact["form_messages"] = build_fdg_form_messages(fact)
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

    async def _generate_outputs(self, tasks: List[Dict[str, Any]]) -> List[Optional[str]]:
        assert self.formalizer is not None
        return await asyncio.to_thread(self.formalizer.generate, [task["messages"] for task in tasks])

    async def _validate_and_apply_one_output(self, task: Dict[str, Any], output: Optional[str]) -> None:
        result = await self._validate_batch_output(task, output)
        async with self.state_lock:
            record = self.records[task["record_id"]]
            fact = record["facts"][task["fact_id"]]
            attempt_num = task["attempt_num"]
            history = list(fact.get("form_attempt_history") or [])
            if result["kind"] == "validated":
                payload = {
                    "lean_code": result["lean_code"],
                    "lean_pass": bool(result["lean_pass"]),
                    "error_msg": [] if result["lean_pass"] else result["error_msg"],
                    "tries": attempt_num,
                    "attempt_history": history,
                }
                success = bool(result["lean_pass"])
                retry_error = f"Lean error: {result['error_msg']}"
            else:
                payload = {
                    "lean_code": result.get("lean_code", ""),
                    "lean_pass": False,
                    "error_msg": result["error_msg"],
                    "tries": attempt_num,
                    "attempt_history": history,
                }
                success = False
                retry_error = f"Error: {result['error_msg']}"

            fact["form_retries_used"] = attempt_num
            fact["formalization"] = payload
            if success:
                fact["form_status"] = "success"
            elif result["kind"] == "token_overflow" or attempt_num >= self.args.formalizer_retries:
                fact["form_status"] = "failed"
            else:
                fact["form_attempt_history"] = history + [payload]
                fact["form_status"] = "pending"
                fact["form_messages"].append(
                    {
                        "role": "user",
                        "content": retry_error + "\n\nBased on the error, please correct the previous response. ",
                    }
                )
                await self._enqueue_form_locked(task["record_id"], task["fact_id"])
            await self._persist_record_locked(task["record_id"])

    async def _run_batch(self, batch_items: List[Tuple[str, str]]) -> None:
        tasks = await self._prepare_batch(batch_items)
        if not tasks:
            return
        outputs = await self._generate_outputs(tasks)
        for task, output in zip(tasks, outputs):
            await self.validation_backpressure.acquire()
            await self.validation_queue.put((task, output))

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

    async def _form_worker(self) -> None:
        while True:
            self._raise_validation_error()
            batch = await self._pop_batch(self.args.form_batch_size)
            if batch:
                print(
                    f"[fdg-form] gpu_batch={len(batch)} pending_validation_items={self._pending_validation_items()} ready_queue={self.form_queue.qsize()}"
                )
                await self._run_batch(batch)
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
            await asyncio.gather(self._form_worker())
        except Exception as exc:
            append_jsonl(self.failed_path, {"created_at": utc_now_iso(), "error": str(exc)})
            raise
        finally:
            await self._cancel_validation_workers()
            if self.lean_server is not None:
                await self.lean_server.aclose()
            if self.formalizer is not None:
                self.formalizer.close()
            self._write_runtime_metrics()

        done = sum(1 for record in self.records.values() if fdg_stage2_record_terminal(record))
        print(f"\n[done] completed={done} out={self.out_path} failed={self.failed_path}")
