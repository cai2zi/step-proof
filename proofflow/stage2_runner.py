from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .lean_check import LeanServer
from .llm_worker import LLMWorkerClient, LLMWorkerConfig
from .stage2_common import (
    FORM_TERMINAL,
    NodeState,
    RecordState,
    StageResult,
    StageTask,
    append_jsonl,
    blocks_children,
    build_form_messages,
    extract_last_lean_block,
    fresh_record_state,
    load_done_ids,
    load_jsonl,
    restore_record_state,
    stage2_checkpoint_payload,
    stage2_final_payload,
    stage2_record_terminal,
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


@dataclass(frozen=True)
class StageSpec:
    name: str
    llm_attr: str
    queue_attr: str
    batch_size_arg: str
    retries_arg: str
    status_key: str
    retries_key: str
    messages_key: str
    history_key: str
    result_key: str
    enqueued_flag: str


FORM_STAGE = StageSpec(
    name="form",
    llm_attr="formalizer",
    queue_attr="form_queue",
    batch_size_arg="form_batch_size",
    retries_arg="formalizer_retries",
    status_key="form_status",
    retries_key="form_retries_used",
    messages_key="form_messages",
    history_key="form_attempt_history",
    result_key="formalization",
    enqueued_flag="_form_enqueued",
)


class Stage2Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.records: Dict[str, RecordState] = {}
        self.form_queue: asyncio.PriorityQueue[Tuple[Tuple[int, int, int], str, str]] = (
            asyncio.PriorityQueue()
        )
        self.state_lock = asyncio.Lock()
        self.lean_semaphore = asyncio.Semaphore(args.lean_check_concurrency)
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir
        self.done_ids: set[str] = set()
        self.formalizer: Optional[LLMWorkerClient] = None
        self.lean_server: Optional[LeanServer] = None
        self.validation_backpressure_limit = self._validation_backpressure_limit()
        self.validation_backpressure = asyncio.Semaphore(self.validation_backpressure_limit)
        self.validation_queue: asyncio.Queue[Tuple[StageSpec, StageTask, Optional[str]]] = (
            asyncio.Queue()
        )
        self.validation_workers: List[asyncio.Task] = []
        self.running_validation_items = 0
        self.validation_error: Optional[BaseException] = None
        self.enqueue_seq = 0
        self.stage_started_perf: Optional[float] = None
        self.gpu_idle_wait_seconds = 0.0
        self.lean_compile_seconds = 0.0
        self.lean_compile_count = 0

    def _validation_backpressure_limit(self) -> int:
        return max(1, self.args.max_pending_validation_batches) * max(
            1,
            getattr(self.args, FORM_STAGE.batch_size_arg),
        )

    def _pending_validation_items(self) -> int:
        return self.validation_queue.qsize() + self.running_validation_items

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
        source_rows = load_jsonl(self.args.infile)
        loaded = 0
        resumed_count = 0
        partial_count = 0
        form_done = 0
        empty_skipped = 0

        for row in source_rows:
            record_id = str(row.get("meta", {}).get("record_id", "")).strip()
            if not record_id:
                continue
            if record_id in self.done_ids:
                resumed_count += 1
                continue

            graph_nodes = row.get("graph", {}).get("nodes", [])
            if not graph_nodes:
                empty_skipped += 1
                append_jsonl(
                    self.failed_path,
                    {
                        "meta": row.get("meta", {}),
                        "error": "Empty graph (no nodes). Skipped by Stage 2.",
                        "created_at": utc_now_iso(),
                    },
                )
                continue

            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if not self.args.no_resume and ckpt_path.is_file():
                record = restore_record_state(
                    json.loads(ckpt_path.read_text(encoding="utf-8"))
                )
                partial_count += 1
                for node in record["nodes"].values():
                    if node["form_status"] in FORM_TERMINAL:
                        form_done += 1
            else:
                record = fresh_record_state(row)
            self.records[record_id] = record
            loaded += 1
            if self.args.limit >= 0 and loaded >= self.args.limit:
                break

        print(f"\n[resume] Fully completed records skipped: {resumed_count}")
        print(f"[resume] Empty graph records skipped: {empty_skipped}")
        print(f"[resume] Partially completed records loaded: {partial_count}")
        print(f"[resume] Nodes already formed: {form_done}")
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
                raise RuntimeError(
                    "--formalizer-chat-template-kwargs-json must be a JSON object"
                )
            chat_template_kwargs = parsed

        self.formalizer = LLMWorkerClient(
            config=LLMWorkerConfig(
                name="formalizer",
                gpus=formalizer_gpus,
                model_path=self.args.formalizer_model_path,
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
            project_path=self.args.mathlib_path,
            backend=self.args.lean_backend,
            pool_size=pool_size,
            temp_root=str(self.args.lean_temp_dir),
        )
        print("[init] stage2 runtime ready.\n")

    def _queue_for(
        self,
        spec: StageSpec,
    ) -> asyncio.PriorityQueue[Tuple[Tuple[int, int, int], str, str]]:
        return getattr(self, spec.queue_attr)

    def _llm_for(self, spec: StageSpec) -> Optional[LLMWorkerClient]:
        return getattr(self, spec.llm_attr)

    def _build_messages_for_stage(
        self,
        spec: StageSpec,
        record: RecordState,
        node: NodeState,
    ) -> List[Dict[str, str]]:
        if spec.name != "form":
            raise ValueError(f"Unsupported Stage 2 stage: {spec.name}")
        return build_form_messages(
            record,
            node,
            include_parent_statement=bool(self.args.include_parent_statement),
            include_parent_nl=bool(self.args.include_parent_nl),
        )

    def _remaining_form_nodes(self, record_id: str) -> int:
        record = self.records[record_id]
        return sum(
            1
            for node in record["nodes"].values()
            if node["form_status"] not in FORM_TERMINAL
        )

    async def _enqueue_stage_locked(
        self,
        spec: StageSpec,
        record_id: str,
        node_id: str,
        *,
        retry: bool,
    ) -> None:
        node = self.records[record_id]["nodes"][node_id]
        if node.get(spec.enqueued_flag):
            return
        node[spec.enqueued_flag] = True
        self.enqueue_seq += 1
        priority = (
            self._remaining_form_nodes(record_id),
            0 if retry else 1,
            self.enqueue_seq,
        )
        await self._queue_for(spec).put((priority, record_id, node_id))

    async def _enqueue_form_locked(
        self,
        record_id: str,
        node_id: str,
        *,
        retry: bool = False,
    ) -> None:
        await self._enqueue_stage_locked(FORM_STAGE, record_id, node_id, retry=retry)

    async def seed_ready_queues(self) -> None:
        async with self.state_lock:
            for record_id, record in self.records.items():
                if stage2_record_terminal(record):
                    await self._persist_record_locked(record_id)
                    continue
                for node_id, node in record["nodes"].items():
                    if node["form_status"] == "pending" and node["blocking_remaining"] == 0:
                        await self._enqueue_form_locked(record_id, node_id)
                if stage2_record_terminal(record):
                    await self._persist_record_locked(record_id)

    async def _persist_record_locked(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if stage2_record_terminal(record):
            payload = stage2_final_payload(record)
            append_jsonl(self.out_path, payload)
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            record["record_status"] = payload["execution"]["record_status"]
            self.done_ids.add(record_id)
            return
        write_json_atomic(
            self.checkpoint_dir / f"{record_id}.json",
            stage2_checkpoint_payload(record),
        )

    async def _mark_children_ready_locked(self, record_id: str, node_id: str) -> None:
        record = self.records[record_id]
        node = record["nodes"][node_id]
        if not blocks_children(node, self.args.id_schema_mode):
            return
        for child_id in node.get("successors") or []:
            child = record["nodes"].get(child_id)
            if child is None or child["form_status"] != "pending":
                continue
            if node_id not in (child.get("blocking_parents") or []):
                continue
            child["blocking_remaining"] = max(0, int(child.get("blocking_remaining", 0)) - 1)
            if child["blocking_remaining"] == 0:
                await self._enqueue_form_locked(record_id, child_id)

    async def _schedule_after_form_locked(self, record_id: str, node_id: str) -> None:
        await self._mark_children_ready_locked(record_id, node_id)

    async def _validate_lean(
        self,
        lean_code: str,
        job_id: str,
    ) -> Tuple[bool, bool, Any]:
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

    async def _validate_batch_output(
        self,
        stage: str,
        task: StageTask,
        output: Optional[str],
    ) -> StageResult:
        if output is None:
            return {
                "kind": "token_overflow",
                "error_msg": "token_overflow",
                "lean_code": "",
            }
        try:
            lean_code = extract_last_lean_block(output)
        except ValueError as e:
            return {
                "kind": "extract_error",
                "error_msg": str(e),
                "lean_code": "",
            }
        lean_pass, lean_verify, error_msg = await self._validate_lean(
            lean_code,
            job_id=f"{stage}_{task['record_id']}_{task['node_id']}_{task['attempt_num']}",
        )
        return {
            "kind": "validated",
            "lean_code": lean_code,
            "lean_pass": lean_pass,
            "lean_verify": lean_verify,
            "error_msg": error_msg,
        }

    async def _pop_batch(
        self,
        queue: asyncio.PriorityQueue[Tuple[Tuple[int, int, int], str, str]],
        batch_size: int,
    ) -> List[Tuple[str, str]]:
        timeout = max(self.args.batch_wait_ms, 1) / 1000.0
        wait_started_perf = time.perf_counter()
        try:
            first_item = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
            return []
        self.gpu_idle_wait_seconds += time.perf_counter() - wait_started_perf
        items = [first_item]
        while len(items) < batch_size:
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return [(record_id, node_id) for _, record_id, node_id in items]

    async def _prepare_stage_batch(
        self,
        spec: StageSpec,
        batch_items: List[Tuple[str, str]],
    ) -> List[StageTask]:
        tasks: List[StageTask] = []
        async with self.state_lock:
            for record_id, node_id in batch_items:
                record = self.records.get(record_id)
                if record is None:
                    continue
                node = record["nodes"].get(node_id)
                if node is None:
                    continue
                node[spec.enqueued_flag] = False
                if node[spec.status_key] != "pending":
                    continue
                node[spec.status_key] = "running"
                if not node[spec.messages_key]:
                    node[spec.messages_key] = self._build_messages_for_stage(spec, record, node)
                attempt_num = int(node.get(spec.retries_key, 0)) + 1
                tasks.append(
                    {
                        "record_id": record_id,
                        "node_id": node_id,
                        "messages": copy.deepcopy(node[spec.messages_key]),
                        "attempt_num": attempt_num,
                    }
                )
        return tasks

    async def _generate_stage_outputs(
        self,
        spec: StageSpec,
        tasks: List[StageTask],
    ) -> List[Optional[str]]:
        llm = self._llm_for(spec)
        assert llm is not None
        outputs = await asyncio.to_thread(
            llm.generate,
            [task["messages"] for task in tasks],
        )
        return outputs

    async def _validate_and_apply_one_output(
        self,
        spec: StageSpec,
        task: StageTask,
        output: Optional[str],
    ) -> None:
        result = await self._validate_batch_output(spec.name, task, output)
        async with self.state_lock:
            record = self.records[task["record_id"]]
            node = record["nodes"][task["node_id"]]
            await self._apply_form_result_locked(record, node, task, result)
            await self._persist_record_locked(task["record_id"])

    async def _run_stage_batch(
        self,
        spec: StageSpec,
        batch_items: List[Tuple[str, str]],
    ) -> None:
        tasks = await self._prepare_stage_batch(spec, batch_items)
        if not tasks:
            return

        outputs = await self._generate_stage_outputs(spec, tasks)
        for task, output in zip(tasks, outputs):
            await self.validation_backpressure.acquire()
            await self.validation_queue.put((spec, task, output))

    def _raise_validation_error(self) -> None:
        if self.validation_error is not None:
            raise RuntimeError("Lean validation worker failed") from self.validation_error

    async def _validation_worker(self, worker_id: int) -> None:
        while True:
            spec, task, output = await self.validation_queue.get()
            self.running_validation_items += 1
            try:
                await self._validate_and_apply_one_output(spec, task, output)
            except Exception as e:
                self.validation_error = e
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

    async def _apply_form_result_locked(
        self,
        record: RecordState,
        node: NodeState,
        task: StageTask,
        result: StageResult,
    ) -> None:
        attempt_num = task["attempt_num"]
        history = list(node.get("form_attempt_history") or [])
        if result["kind"] == "validated":
            payload = {
                "lean_code": result["lean_code"],
                "lean_pass": bool(result["lean_pass"]),
                "error_msg": [] if result["lean_pass"] else result["error_msg"],
                "tries": attempt_num,
                "attempt_history": history,
                "dependency_context_block": node.get("dependency_context_block", ""),
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
                "dependency_context_block": node.get("dependency_context_block", ""),
            }
            success = False
            retry_error = f"Error: {result['error_msg']}"

        node["form_retries_used"] = attempt_num
        node["formalization"] = payload
        if success:
            node["form_status"] = "success"
            await self._schedule_after_form_locked(record["meta"]["record_id"], node["id"])
            return

        if result["kind"] == "token_overflow" or attempt_num >= self.args.formalizer_retries:
            node["form_status"] = "failed"
            await self._schedule_after_form_locked(record["meta"]["record_id"], node["id"])
            return

        node["form_attempt_history"] = history + [payload]
        node["form_status"] = "pending"
        node["form_messages"].append(
            {
                "role": "user",
                "content": (
                    retry_error
                    + "\n\nBased on the error, please correct the previous response. "
                ),
            }
        )
        await self._enqueue_form_locked(
            record["meta"]["record_id"],
            node["id"],
            retry=True,
        )

    async def _form_worker(self) -> None:
        while True:
            self._raise_validation_error()
            batch = await self._pop_batch(
                self._queue_for(FORM_STAGE),
                getattr(self.args, FORM_STAGE.batch_size_arg),
            )
            if batch:
                print(
                    f"[form] gpu_batch={len(batch)} "
                    f"pending_validation_items={self._pending_validation_items()} "
                    f"ready_queue={self._queue_for(FORM_STAGE).qsize()}"
                )
                await self._run_stage_batch(FORM_STAGE, batch)
                continue
            async with self.state_lock:
                records_terminal = all(
                    stage2_record_terminal(record) for record in self.records.values()
                )
                if records_terminal and self._pending_validation_items() == 0:
                    return
            await asyncio.sleep(max(self.args.batch_wait_ms, 1) / 1000.0)

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
        await self.seed_ready_queues()
        self._start_validation_workers()
        try:
            await asyncio.gather(
                self._form_worker(),
            )
        except Exception as e:
            append_jsonl(
                self.failed_path,
                {
                    "created_at": utc_now_iso(),
                    "error": str(e),
                },
            )
            raise
        finally:
            await self._cancel_validation_workers()
            if self.lean_server is not None:
                await self.lean_server.aclose()
            if self.formalizer is not None:
                self.formalizer.close()
            self._write_runtime_metrics()

        done = sum(1 for record in self.records.values() if stage2_record_terminal(record))
        print(f"\n[done] completed={done} out={self.out_path} failed={self.failed_path}")

    def _write_runtime_metrics(self) -> None:
        if self.args.metrics_out is None:
            return
        total_seconds = 0.0
        if self.stage_started_perf is not None:
            total_seconds = time.perf_counter() - self.stage_started_perf
        payload = {
            "stage": "stage2",
            "total_execution_seconds": round(total_seconds, 6),
            "lean_compile": {
                "total_seconds": round(self.lean_compile_seconds, 6),
                "node_count": self.lean_compile_count,
                "avg_seconds_per_node": round(
                    self.lean_compile_seconds / self.lean_compile_count,
                    6,
                )
                if self.lean_compile_count
                else None,
            },
            "gpu_idle_wait": {
                "total_seconds": round(self.gpu_idle_wait_seconds, 6),
                "ratio_of_total": round(
                    self.gpu_idle_wait_seconds / total_seconds,
                    6,
                )
                if total_seconds > 0
                else None,
            },
        }
        write_json_atomic(self.args.metrics_out, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 (batch + local vLLM): form from graph-v1 JSONL.",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "graphs.jsonl",
        help="Stage 1 graph-v1 JSONL",
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
        choices=["subprocess", "persistent_lsp"],
    )
    parser.add_argument("--lean-check-concurrency", type=int, default=16)
    parser.add_argument("--lean-worker-pool-size", type=int, default=0)
    parser.add_argument(
        "--include-parent-statement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include parent-node statement text in Stage 2 dependency context.",
    )
    parser.add_argument(
        "--include-parent-nl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include parent-node natural_language text in Stage 2 dependency context.",
    )
    parser.add_argument(
        "--lean-temp-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "lean_jobs",
    )

    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--formalizer-gpus", default=DEFAULT_FORMALIZER_GPUS)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--id-schema-mode", default="calc")
    parser.add_argument("--batch-wait-ms", type=int, default=200)
    parser.add_argument(
        "--max-pending-validation-batches",
        type=int,
        default=4,
        help="Max generated form batches allowed to wait for Lean validation.",
    )

    parser.add_argument("--formalizer-model-path", default=DEFAULT_FORMALIZER_MODEL_PATH)
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
        help=(
            'JSON object for tokenizer.apply_chat_template '
            '(default: {"enable_thinking": false})'
        ),
    )
    parser.add_argument("--formalizer-retries", type=int, default=3)
    parser.add_argument("--form-batch-size", type=int, default=64)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON file for stage runtime metrics.",
    )
    return parser
