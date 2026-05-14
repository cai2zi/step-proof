from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .fdg_graph import FDG_OUTPUT_TRUNCATED_RETRY_HINT
from .fdg_stage_common import (
    PROVE_TERMINAL,
    build_fdg_prove_messages,
    fdg_stage3_checkpoint_payload,
    fdg_stage3_final_payload,
    fdg_stage3_record_terminal,
    fresh_fdg_stage3_record_state,
    restore_fdg_stage3_record_state,
)
from .lean_check import LeanServer
from .llm_worker import LLMWorkerConfig, LLMWorkerPool
from .runtime_common import (
    append_jsonl,
    extract_last_lean_block,
    load_done_ids,
    load_jsonl,
    utc_now_iso,
    write_json_atomic,
)

load_dotenv()

DEFAULT_PROVER_MODEL_PATH = os.getenv(
    "PROVER_MODEL_PATH", "/data/czx/models/Goedel-Prover-V2-8B"
)
DEFAULT_GPUS = os.getenv(
    "STAGE3_GPUS",
    os.getenv("STAGE2_GPUS", os.getenv("GRAPH_GPUS", "0,1,2,3")),
)
DEFAULT_PROVER_GPUS = os.getenv("PROVER_GPUS", "")
DEFAULT_PROVER_TP = int(os.getenv("PROVER_TP", "2"))
DEFAULT_MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")
DEFAULT_LEAN_BACKEND = os.getenv("LEAN_BACKEND", "subprocess")


class FDGStage3Runner:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        lean_server: Optional[LeanServer] = None,
        owned_lean_server: Optional[bool] = None,
    ) -> None:
        self.args = args
        self.records: Dict[str, Dict[str, Any]] = {}
        self.prove_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.state_lock = asyncio.Lock()
        self.lean_semaphore = asyncio.Semaphore(args.lean_check_concurrency)
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir
        self.done_ids: set[str] = set()
        self.provers: Optional[LLMWorkerPool] = None
        self.lean_server: Optional[LeanServer] = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.validation_backpressure_limit = max(1, args.max_pending_validation_batches) * max(1, args.prove_batch_size)
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

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
        source_rows = load_jsonl(self.args.infile)
        loaded = resumed_count = partial_count = prove_done = empty_skipped = 0

        for row in source_rows:
            record_id = str(row.get("meta", {}).get("record_id", "")).strip()
            if not record_id:
                continue
            if record_id in self.done_ids:
                resumed_count += 1
                continue

            result_facts = (row.get("results") or {}).get("facts", [])
            if not result_facts:
                empty_skipped += 1
                append_jsonl(
                    self.failed_path,
                    {
                        "meta": row.get("meta", {}),
                        "error": "Empty FDG stage2 results (no results.facts). Skipped by Stage 3.",
                        "created_at": utc_now_iso(),
                    },
                )
                continue

            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if not self.args.no_resume and ckpt_path.is_file():
                record = restore_fdg_stage3_record_state(
                    json.loads(ckpt_path.read_text(encoding="utf-8"))
                )
                partial_count += 1
                for fact in record["facts"].values():
                    if fact.get("prove_status") in PROVE_TERMINAL:
                        prove_done += 1
            else:
                record = fresh_fdg_stage3_record_state(row)

            self._ensure_prompt_meta(record)
            self.records[record_id] = record
            loaded += 1
            if self.args.limit >= 0 and loaded >= self.args.limit:
                break

        print(f"\n[resume] Fully completed records skipped: {resumed_count}")
        print(f"[resume] Empty FDG results skipped: {empty_skipped}")
        print(f"[resume] Partially completed records loaded: {partial_count}")
        print(f"[resume] Facts already proved/skipped: {prove_done}")
        print(f"[resume] Total pending records to process: {loaded}\n")

    def _ensure_prompt_meta(self, record: Dict[str, Any]) -> None:
        prompt = str(getattr(self.args, "prover_prompt", "prove") or "prove")
        existing = str((record.get("meta") or {}).get("prover_prompt") or "")
        if existing and existing != prompt:
            rid = (record.get("meta") or {}).get("record_id", "<unknown>")
            raise RuntimeError(
                f"Record {rid} checkpoint was created with prover_prompt={existing!r}, "
                f"but current config uses {prompt!r}. Use a new exp.name or disable resume."
            )
        record.setdefault("meta", {})["prover_prompt"] = prompt

    def _resolve_prover_gpus(self) -> str:
        if self.args.prover_gpus:
            return self.args.prover_gpus
        all_devices = [device.strip() for device in self.args.gpus.split(",") if device.strip()]
        total_needed = self.args.prover_instances * self.args.prover_tensor_parallel_size
        if len(all_devices) < total_needed:
            raise RuntimeError(
                "Not enough GPUs in --gpus to derive the prover worker. "
                f"Need {total_needed}, got {len(all_devices)} from {self.args.gpus!r}."
            )
        return ",".join(all_devices[:total_needed])

    async def init_runtime(self) -> None:
        prover_gpus = self._resolve_prover_gpus()
        print(
            "[init] loading prover",
            self.args.prover_model_path,
            (
                f"(instances={self.args.prover_instances}, "
                f"tp={self.args.prover_tensor_parallel_size}, gpus={prover_gpus}) ..."
            ),
        )
        chat_template_kwargs: Dict[str, Any] = {"enable_thinking": False}
        if self.args.prover_chat_template_kwargs_json:
            parsed = json.loads(self.args.prover_chat_template_kwargs_json)
            if not isinstance(parsed, dict):
                raise RuntimeError("--prover-chat-template-kwargs-json must be a JSON object")
            chat_template_kwargs = parsed

        self.provers = LLMWorkerPool(
            base_config=LLMWorkerConfig(
                name="prover",
                gpus="",
                model_path=self.args.prover_model_path or DEFAULT_PROVER_MODEL_PATH,
                tensor_parallel_size=self.args.prover_tensor_parallel_size,
                max_tokens=self.args.prover_max_tokens,
                temperature=self.args.prover_temperature,
                token_limit=self.args.prover_token_limit,
                dtype=self.args.dtype,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                top_p=self.args.prover_top_p,
                presence_penalty=self.args.prover_presence_penalty,
                frequency_penalty=self.args.prover_frequency_penalty,
                seed=self.args.prover_seed,
                top_k=self.args.prover_top_k,
                chat_template_kwargs=chat_template_kwargs,
            ),
            instances=self.args.prover_instances,
            gpus=prover_gpus,
            startup_timeout=self.args.prover_startup_timeout,
            parallel_startup=self.args.prover_parallel_startup,
            startup_stagger_seconds=self.args.prover_startup_stagger_seconds,
        )
        if self.lean_server is None:
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
            print("[init] fdg stage3 using shared Lean runtime.")
        print("[init] fdg stage3 runtime ready.\n")

    async def seed_ready_queue(self) -> None:
        async with self.state_lock:
            for record_id, record in self.records.items():
                if fdg_stage3_record_terminal(record):
                    await self._persist_record_locked(record_id)
                    continue
                for fact_id, fact in record["facts"].items():
                    if fact.get("prove_status") == "pending":
                        await self._enqueue_prove_locked(record_id, fact_id)

    async def _persist_record_locked(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if fdg_stage3_record_terminal(record):
            payload = fdg_stage3_final_payload(record)
            append_jsonl(self.out_path, payload)
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            record["record_status"] = payload["execution"]["record_status"]
            self.done_ids.add(record_id)
            return
        write_json_atomic(self.checkpoint_dir / f"{record_id}.json", fdg_stage3_checkpoint_payload(record))

    async def _enqueue_prove_locked(self, record_id: str, fact_id: str) -> None:
        fact = self.records[record_id]["facts"][fact_id]
        if fact.get("_prove_enqueued"):
            return
        fact["_prove_enqueued"] = True
        await self.prove_queue.put((record_id, fact_id))

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
        lean_pass, lean_verify, error_msg = await self._validate_lean(
            lean_code,
            job_id=f"fdg_stage3_{task['record_id']}_{task['fact_id']}_{task['attempt_num']}",
        )
        return {
            "kind": "validated",
            "lean_code": lean_code,
            "lean_pass": lean_pass,
            "lean_verify": lean_verify,
            "error_msg": error_msg,
        }

    async def _pop_batch(self, batch_size: int) -> List[Tuple[str, str]]:
        timeout = max(self.args.batch_wait_ms, 1) / 1000.0
        wait_started_perf = time.perf_counter()
        try:
            first_item = await asyncio.wait_for(self.prove_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
            return []
        self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
        items = [first_item]
        while len(items) < batch_size:
            try:
                items.append(self.prove_queue.get_nowait())
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
                fact["_prove_enqueued"] = False
                if fact.get("prove_status") != "pending":
                    continue
                fact["prove_status"] = "running"
                if not fact.get("prove_messages"):
                    fact["prove_messages"] = build_fdg_prove_messages(
                        fact,
                        prompt_name=self.args.prover_prompt,
                    )
                attempt_num = int(fact.get("prove_retries_used", 0)) + 1
                tasks.append(
                    {
                        "record_id": record_id,
                        "fact_id": fact_id,
                        "messages": copy.deepcopy(fact["prove_messages"]),
                        "attempt_num": attempt_num,
                    }
                )
        return tasks

    async def _generate_outputs(self, worker_id: int, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assert self.provers is not None
        return await asyncio.to_thread(
            self.provers[worker_id].generate_with_metadata,
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
            if result["kind"] == "validated":
                payload = {
                    "lean_code": result["lean_code"],
                    "lean_pass": bool(result["lean_pass"]),
                    "lean_verify": bool(result["lean_verify"]),
                    "error_msg": result["error_msg"],
                    "tries": attempt_num,
                    "conversation": conversation,
                }
                success = bool(result["lean_verify"])
                retry_error = f"Lean error/warnings: {result['error_msg']}"
            else:
                payload = {
                    "lean_code": result.get("lean_code", ""),
                    "lean_pass": False,
                    "lean_verify": False,
                    "error_msg": result["error_msg"],
                    "tries": attempt_num,
                    "conversation": conversation,
                }
                success = False
                retry_error = f"Error: {result['error_msg']}"

            fact["prove_retries_used"] = attempt_num
            fact["solved_lemma"] = payload
            if success:
                fact["prove_status"] = "success"
            elif result["kind"] == "token_overflow" or attempt_num > self.args.prover_retries:
                fact["prove_status"] = "failed"
            else:
                fact["prove_status"] = "pending"
                fact["prove_messages"] = conversation
                fact["prove_messages"].append(
                    {
                        "role": "user",
                        "content": retry_error + "\n\n Based on these errors, please correct the previous response. ",
                    }
                )
                await self._enqueue_prove_locked(task["record_id"], task["fact_id"])
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
            raise RuntimeError("FDG stage3 validation worker failed") from self.validation_error

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

    async def _prove_worker(self, worker_id: int) -> None:
        while True:
            self._raise_validation_error()
            batch = await self._pop_batch(self.args.prove_batch_size)
            if batch:
                print(
                    f"[fdg-prove:{worker_id}] gpu_batch={len(batch)} "
                    f"pending_validation_items={self._pending_validation_items()} "
                    f"ready_queue={self.prove_queue.qsize()}"
                )
                await self._run_batch(worker_id, batch)
                continue
            async with self.state_lock:
                records_terminal = all(fdg_stage3_record_terminal(record) for record in self.records.values())
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
            "stage": "stage3",
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
            worker_count = len(self.provers) if self.provers is not None else 1
            await asyncio.gather(
                *(self._prove_worker(worker_id) for worker_id in range(worker_count))
            )
        except Exception as exc:
            append_jsonl(self.failed_path, {"created_at": utc_now_iso(), "error": str(exc)})
            raise
        finally:
            await self._cancel_validation_workers()
            if self.owned_lean_server and self.lean_server is not None:
                await self.lean_server.aclose()
            if self.provers is not None:
                self.provers.close()
            self._write_runtime_metrics()

        done = sum(1 for record in self.records.values() if fdg_stage3_record_terminal(record))
        print(f"\n[done] completed={done} out={self.out_path} failed={self.failed_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FDG Stage 3 (batch + local vLLM): prove formalized FDG facts.",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage2_results.jsonl",
        help="Stage 2 FDG formalization JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage3_results.jsonl",
        help="Stage 3 final results JSONL",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage3_failed.jsonl",
        help="Stage 3 fatal failures JSONL",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "stage3_ckpt",
        help="Checkpoint directory for partial record states",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Max NEW records to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior outputs/checkpoints")

    parser.add_argument("--mathlib-path", default=DEFAULT_MATHLIB_PATH)
    parser.add_argument(
        "--lean-backend",
        default=DEFAULT_LEAN_BACKEND,
        choices=["subprocess", "persistent_lsp"],
    )
    parser.add_argument("--lean-check-concurrency", type=int, default=16)
    parser.add_argument("--lean-worker-pool-size", type=int, default=0)
    parser.add_argument(
        "--lean-temp-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "lean_jobs",
    )

    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--prover-gpus", default=DEFAULT_PROVER_GPUS)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-wait-ms", type=int, default=200)
    parser.add_argument(
        "--max-pending-validation-batches",
        type=int,
        default=4,
        help="Max generated prove batches allowed to wait for Lean validation.",
    )

    parser.add_argument("--prover-model-path", default=DEFAULT_PROVER_MODEL_PATH)
    parser.add_argument("--prover-instances", type=int, default=1)
    parser.add_argument("--prover-parallel-startup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prover-startup-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--prover-startup-timeout", type=int, default=1800)
    parser.add_argument("--prover-tensor-parallel-size", type=int, default=DEFAULT_PROVER_TP)
    parser.add_argument("--prover-max-tokens", type=int, default=8192)
    parser.add_argument("--prover-token-limit", type=int, default=32768)
    parser.add_argument("--prover-temperature", type=float, default=0.0)
    parser.add_argument("--prover-top-p", type=float, default=1.0)
    parser.add_argument("--prover-presence-penalty", type=float, default=0.0)
    parser.add_argument("--prover-frequency-penalty", type=float, default=0.0)
    parser.add_argument("--prover-seed", type=int, default=42)
    parser.add_argument("--prover-top-k", type=int, default=20)
    parser.add_argument(
        "--prover-chat-template-kwargs-json",
        default=None,
        help='JSON object for tokenizer.apply_chat_template (default: {"enable_thinking": false})',
    )
    parser.add_argument(
        "--prover-prompt",
        default="prove",
        help="Prover prompt file stem under prompts/{system,user}.",
    )
    parser.add_argument(
        "--prover-retries",
        type=int,
        default=3,
        help="Maximum retry rounds after the initial prover attempt.",
    )
    parser.add_argument("--prove-batch-size", type=int, default=64)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON file for stage runtime metrics.",
    )
    return parser
