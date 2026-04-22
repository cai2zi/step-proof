from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
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
    build_prove_messages,
    checkpoint_payload,
    empty_solver,
    extract_last_lean_block,
    final_payload,
    fresh_record_state,
    load_done_ids,
    load_jsonl,
    record_terminal,
    restore_record_state,
    should_prove,
    utc_now_iso,
    write_json_atomic,
)

load_dotenv()

DEFAULT_FORMALIZER_MODEL_PATH = os.getenv(
    "FORMALIZER_MODEL_PATH", "/data/czx/models/Goedel-Formalizer-V2-8B"
)
DEFAULT_PROVER_MODEL_PATH = os.getenv(
    "PROVER_MODEL_PATH", "/data/czx/models/Goedel-Prover-V2-8B"
)
DEFAULT_GPUS = os.getenv("STAGE2_GPUS", os.getenv("GRAPH_GPUS", "0,1,2,3"))
DEFAULT_FORMALIZER_GPUS = os.getenv("FORMALIZER_GPUS", "")
DEFAULT_PROVER_GPUS = os.getenv("PROVER_GPUS", "")
DEFAULT_FORMALIZER_TP = int(os.getenv("FORMALIZER_TP", "2"))
DEFAULT_PROVER_TP = int(os.getenv("PROVER_TP", "2"))
DEFAULT_MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")


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
PROVE_STAGE = StageSpec(
    name="prove",
    llm_attr="prover",
    queue_attr="prove_queue",
    batch_size_arg="prove_batch_size",
    retries_arg="prover_retries",
    status_key="prove_status",
    retries_key="prove_retries_used",
    messages_key="prove_messages",
    history_key="prove_attempt_history",
    result_key="solved_lemma",
    enqueued_flag="_prove_enqueued",
)


class Stage2Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.records: Dict[str, RecordState] = {}
        self.form_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.prove_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.state_lock = asyncio.Lock()
        self.lean_semaphore = asyncio.Semaphore(args.lean_check_concurrency)
        self.out_path = args.out
        self.failed_path = args.failed
        self.checkpoint_dir = args.checkpoint_dir
        self.done_ids: set[str] = set()
        self.formalizer: Optional[LLMWorkerClient] = None
        self.prover: Optional[LLMWorkerClient] = None
        self.lean_server: Optional[LeanServer] = None

    def load_records(self) -> None:
        self.done_ids = set() if self.args.no_resume else load_done_ids(self.out_path)
        source_rows = load_jsonl(self.args.infile)
        loaded = 0
        resumed_count = 0
        partial_count = 0
        form_done = 0
        prove_done = 0
        empty_skipped = 0

        for row in source_rows:
            record_id = str(row.get("meta", {}).get("record_id", "")).strip()
            if not record_id:
                continue
            if record_id in self.done_ids:
                resumed_count += 1
                continue

            # Protection A: Skip empty graphs
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
                    if node["prove_status"] in {"success", "failed", "skipped"}:
                        prove_done += 1
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
        print(f"[resume] Nodes already proved/skipped: {prove_done}")
        print(f"[resume] Total pending records to process: {loaded}\n")

    def _resolve_stage_gpus(self) -> tuple[str, str]:
        if self.args.formalizer_gpus and self.args.prover_gpus:
            return self.args.formalizer_gpus, self.args.prover_gpus

        all_devices = [device.strip() for device in self.args.gpus.split(",") if device.strip()]
        total_needed = (
            self.args.formalizer_tensor_parallel_size
            + self.args.prover_tensor_parallel_size
        )
        if len(all_devices) < total_needed:
            raise RuntimeError(
                "Not enough GPUs in --gpus to derive formalizer/prover workers. "
                f"Need {total_needed}, got {len(all_devices)} from {self.args.gpus!r}."
            )

        form_gpus = self.args.formalizer_gpus or ",".join(
            all_devices[: self.args.formalizer_tensor_parallel_size]
        )
        prover_start = self.args.formalizer_tensor_parallel_size
        prove_slice = all_devices[
            prover_start : prover_start + self.args.prover_tensor_parallel_size
        ]
        prover_gpus = self.args.prover_gpus or ",".join(prove_slice)
        return form_gpus, prover_gpus

    async def init_runtime(self) -> None:
        formalizer_gpus, prover_gpus = self._resolve_stage_gpus()
        print(
            "[init] loading formalizer",
            self.args.formalizer_model_path,
            f"(tp={self.args.formalizer_tensor_parallel_size}, gpus={formalizer_gpus}) ...",
        )
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
            )
        )
        print(
            "[init] loading prover",
            self.args.prover_model_path,
            f"(tp={self.args.prover_tensor_parallel_size}, gpus={prover_gpus}) ...",
        )
        self.prover = LLMWorkerClient(
            config=LLMWorkerConfig(
                name="prover",
                gpus=prover_gpus,
                model_path=self.args.prover_model_path,
                tensor_parallel_size=self.args.prover_tensor_parallel_size,
                max_tokens=self.args.prover_max_tokens,
                temperature=self.args.prover_temperature,
                token_limit=self.args.prover_token_limit,
                dtype=self.args.dtype,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
            )
        )
        self.lean_server = LeanServer(project_path=self.args.mathlib_path)
        print("[init] stage2 runtime ready.\n")

    def _prove_ready(self, node: NodeState) -> bool:
        if node["prove_status"] != "pending":
            return False
        if node["form_status"] not in FORM_TERMINAL:
            return False
        if not should_prove(node, self.args.id_schema_mode):
            return False
        return bool((node.get("formalization") or {}).get("lean_code"))

    def _queue_for(self, spec: StageSpec) -> asyncio.Queue[Tuple[str, str]]:
        return getattr(self, spec.queue_attr)

    def _llm_for(self, spec: StageSpec) -> Optional[LLMWorkerClient]:
        return getattr(self, spec.llm_attr)

    def _build_messages_for_stage(
        self,
        spec: StageSpec,
        record: RecordState,
        node: NodeState,
    ) -> List[Dict[str, str]]:
        if spec.name == "form":
            return build_form_messages(record, node)
        return build_prove_messages(node)

    async def _enqueue_stage_locked(
        self,
        spec: StageSpec,
        record_id: str,
        node_id: str,
    ) -> None:
        node = self.records[record_id]["nodes"][node_id]
        if node.get(spec.enqueued_flag):
            return
        node[spec.enqueued_flag] = True
        await self._queue_for(spec).put((record_id, node_id))

    async def _enqueue_form_locked(self, record_id: str, node_id: str) -> None:
        await self._enqueue_stage_locked(FORM_STAGE, record_id, node_id)

    async def _enqueue_prove_locked(self, record_id: str, node_id: str) -> None:
        await self._enqueue_stage_locked(PROVE_STAGE, record_id, node_id)

    async def seed_ready_queues(self) -> None:
        async with self.state_lock:
            for record_id, record in self.records.items():
                if record_terminal(record):
                    await self._persist_record_locked(record_id)
                    continue
                for node_id, node in record["nodes"].items():
                    if node["form_status"] == "pending" and node["blocking_remaining"] == 0:
                        await self._enqueue_form_locked(record_id, node_id)
                    elif (
                        node["prove_status"] == "pending"
                        and node["form_status"] in FORM_TERMINAL
                        and not (node.get("formalization") or {}).get("lean_code")
                    ):
                        node["prove_status"] = "skipped"
                        node["solved_lemma"] = empty_solver(skipped=True)
                    elif self._prove_ready(node):
                        await self._enqueue_prove_locked(record_id, node_id)
                if record_terminal(record):
                    await self._persist_record_locked(record_id)

    async def _persist_record_locked(self, record_id: str) -> None:
        if record_id in self.done_ids:
            return
        record = self.records[record_id]
        if record_terminal(record):
            payload = final_payload(record)
            append_jsonl(self.out_path, payload)
            ckpt_path = self.checkpoint_dir / f"{record_id}.json"
            if ckpt_path.exists():
                ckpt_path.unlink()
            record["record_status"] = payload["execution"]["record_status"]
            self.done_ids.add(record_id)
            return
        write_json_atomic(
            self.checkpoint_dir / f"{record_id}.json",
            checkpoint_payload(record),
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
        record = self.records[record_id]
        node = record["nodes"][node_id]
        if self._prove_ready(node):
            await self._enqueue_prove_locked(record_id, node_id)
        elif node["prove_status"] == "pending" and not (node.get("formalization") or {}).get("lean_code"):
            node["prove_status"] = "skipped"
            node["solved_lemma"] = empty_solver(skipped=True)
        await self._mark_children_ready_locked(record_id, node_id)

    async def _validate_lean(
        self,
        lean_code: str,
        job_id: str,
    ) -> Tuple[bool, bool, Any]:
        assert self.lean_server is not None
        async with self.lean_semaphore:
            return await self.lean_server.check_lean_string_async(
                lean_code,
                temp_root=str(self.args.lean_temp_dir),
                job_id=job_id,
            )

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
        queue: asyncio.Queue[Tuple[str, str]],
        batch_size: int,
    ) -> List[Tuple[str, str]]:
        timeout = max(self.args.batch_wait_ms, 1) / 1000.0
        try:
            first_item = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return []
        items = [first_item]
        while len(items) < batch_size:
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def _run_stage_batch(
        self,
        spec: StageSpec,
        batch_items: List[Tuple[str, str]],
    ) -> None:
        llm = self._llm_for(spec)
        assert llm is not None
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

        if not tasks:
            return

        outputs = await asyncio.to_thread(
            llm.generate,
            [task["messages"] for task in tasks],
        )
        results = await asyncio.gather(
            *[
                self._validate_batch_output(spec.name, task, output)
                for task, output in zip(tasks, outputs)
            ]
        )

        dirty_record_ids: set[str] = set()
        async with self.state_lock:
            for task, result in zip(tasks, results):
                record = self.records[task["record_id"]]
                node = record["nodes"][task["node_id"]]
                dirty_record_ids.add(task["record_id"])
                if spec.name == "form":
                    await self._apply_form_result_locked(record, node, task, result)
                else:
                    await self._apply_prove_result_locked(record, node, task, result)
            for record_id in sorted(dirty_record_ids):
                await self._persist_record_locked(record_id)

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
        await self._enqueue_form_locked(record["meta"]["record_id"], node["id"])

    async def _apply_prove_result_locked(
        self,
        record: RecordState,
        node: NodeState,
        task: StageTask,
        result: StageResult,
    ) -> None:
        attempt_num = task["attempt_num"]
        history = list(node.get("prove_attempt_history") or [])
        if result["kind"] == "validated":
            payload = {
                "lean_code": result["lean_code"],
                "lean_pass": bool(result["lean_pass"]),
                "lean_verify": bool(result["lean_verify"]),
                "error_msg": result["error_msg"],
                "tries": attempt_num,
                "attempt_history": history,
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
                "attempt_history": history,
            }
            success = False
            retry_error = f"Error: {result['error_msg']}"

        node["prove_retries_used"] = attempt_num
        node["solved_lemma"] = payload
        if success:
            node["prove_status"] = "success"
            return

        if result["kind"] == "token_overflow" or attempt_num >= self.args.prover_retries:
            node["prove_status"] = "failed"
            return

        node["prove_attempt_history"] = history + [payload]
        node["prove_status"] = "pending"
        node["prove_messages"].append(
            {
                "role": "user",
                "content": (
                    retry_error
                    + "\n\n Based on these errors, please correct the previous response. "
                ),
            }
        )
        await self._enqueue_prove_locked(record["meta"]["record_id"], node["id"])

    async def _form_worker(self) -> None:
        while True:
            batch = await self._pop_batch(
                self._queue_for(FORM_STAGE),
                getattr(self.args, FORM_STAGE.batch_size_arg),
            )
            if batch:
                print(f"[form] batch={len(batch)}")
                await self._run_stage_batch(FORM_STAGE, batch)
                continue
            async with self.state_lock:
                if all(record_terminal(record) for record in self.records.values()):
                    return

    async def _prove_worker(self) -> None:
        while True:
            batch = await self._pop_batch(
                self._queue_for(PROVE_STAGE),
                getattr(self.args, PROVE_STAGE.batch_size_arg),
            )
            if batch:
                print(f"[prove] batch={len(batch)}")
                await self._run_stage_batch(PROVE_STAGE, batch)
                continue
            async with self.state_lock:
                if all(record_terminal(record) for record in self.records.values()):
                    return

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
            print("[done] no pending records.")
            return

        await self.init_runtime()
        await self.seed_ready_queues()
        try:
            await asyncio.gather(
                self._form_worker(),
                self._prove_worker(),
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
            if self.formalizer is not None:
                self.formalizer.close()
            if self.prover is not None:
                self.prover.close()

        done = sum(1 for record in self.records.values() if record_terminal(record))
        print(f"\n[done] completed={done} out={self.out_path} failed={self.failed_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 (batch + local vLLM): form + prove from graph-v1 JSONL.",
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
    parser.add_argument("--lean-check-concurrency", type=int, default=16)
    parser.add_argument(
        "--lean-temp-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calc_runs" / "lean_jobs",
    )

    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--formalizer-gpus", default=DEFAULT_FORMALIZER_GPUS)
    parser.add_argument("--prover-gpus", default=DEFAULT_PROVER_GPUS)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--id-schema-mode", default="calc")
    parser.add_argument("--batch-wait-ms", type=int, default=200)

    parser.add_argument("--formalizer-model-path", default=DEFAULT_FORMALIZER_MODEL_PATH)
    parser.add_argument("--formalizer-tensor-parallel-size", type=int, default=DEFAULT_FORMALIZER_TP)
    parser.add_argument("--formalizer-max-tokens", type=int, default=8192)
    parser.add_argument("--formalizer-token-limit", type=int, default=32768)
    parser.add_argument("--formalizer-temperature", type=float, default=0.2)
    parser.add_argument("--formalizer-retries", type=int, default=3)
    parser.add_argument("--form-batch-size", type=int, default=64)

    parser.add_argument("--prover-model-path", default=DEFAULT_PROVER_MODEL_PATH)
    parser.add_argument("--prover-tensor-parallel-size", type=int, default=DEFAULT_PROVER_TP)
    parser.add_argument("--prover-max-tokens", type=int, default=8192)
    parser.add_argument("--prover-token-limit", type=int, default=32768)
    parser.add_argument("--prover-temperature", type=float, default=0.2)
    parser.add_argument("--prover-retries", type=int, default=3)
    parser.add_argument("--prove-batch-size", type=int, default=64)
    return parser
