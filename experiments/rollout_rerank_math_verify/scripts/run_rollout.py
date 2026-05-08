#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common import exp_dir, load_config, read_done_ids
from proofflow.llm_worker import split_gpu_groups


class ProgressBar:
    def __init__(self, *, total: int, desc: str, enabled: bool = True) -> None:
        self.total = int(total)
        self.current = 0
        self._bar = None
        self._enabled = bool(enabled)
        if not self._enabled:
            return
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=self.total, desc=desc, unit="prompt", dynamic_ncols=True)
        except Exception:
            print(f"[progress] {desc}: 0/{self.total}", flush=True)

    def update(self, count: int) -> None:
        self.current += int(count)
        if not self._enabled:
            return
        if self._bar is not None:
            self._bar.update(int(count))
        else:
            print(f"[progress] rollout: {self.current}/{self.total}", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


@dataclass(frozen=True)
class RolloutWorkerConfig:
    name: str
    gpus: str
    model_path: str
    tensor_parallel_size: int
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_tokens: int
    n: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


def _worker_main(
    config_dict: Dict[str, Any],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    try:
        config = RolloutWorkerConfig(**config_dict)
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            n=config.n,
            seed=config.seed,
        )
        response_queue.put({"type": "ready", "name": config.name, "gpus": config.gpus})

        while True:
            request = request_queue.get()
            req_type = request.get("type")
            if req_type == "shutdown":
                return
            if req_type != "generate":
                response_queue.put({"type": "error", "error": f"unknown request: {req_type}"})
                continue
            try:
                items = request["items"]
                prompts = [
                    tokenizer.apply_chat_template(
                        item["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        **config.chat_template_kwargs,
                    )
                    for item in items
                ]
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                results = []
                for item, output in zip(items, outputs):
                    rec = {
                        "id": item["id"],
                        "source": item["source"],
                        "question": item["question"],
                        "gold": item["gold"],
                    }
                    for idx, completion in enumerate(output.outputs, start=1):
                        rec[f"response_{idx}"] = completion.text
                        rec[f"finish_reason_{idx}"] = completion.finish_reason
                    results.append(rec)
                response_queue.put({"type": "result", "results": results})
            except Exception:
                response_queue.put({"type": "error", "error": traceback.format_exc()})
    except Exception:
        response_queue.put({"type": "startup_error", "error": traceback.format_exc()})


class RolloutWorkerClient:
    def __init__(
        self,
        config: RolloutWorkerConfig,
        startup_timeout: int = 1800,
        wait_ready: bool = True,
    ) -> None:
        self.config = config
        self._ready = False
        self._ctx = mp.get_context("spawn")
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(asdict(config), self._request_queue, self._response_queue),
            daemon=False,
            name=f"rollout-worker-{config.name}",
        )
        self._process.start()
        if wait_ready:
            self.wait_until_ready(startup_timeout)

    def wait_until_ready(self, timeout: int = 1800) -> None:
        if self._ready:
            return
        try:
            message = self._response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self.close(force=True)
            raise RuntimeError(f"{self.config.name} did not become ready") from exc
        if message.get("type") == "ready":
            self._ready = True
            return
        self.close(force=True)
        raise RuntimeError(f"{self.config.name} failed during startup:\n{message.get('error', message)}")

    def generate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._request_queue.put({"type": "generate", "items": items})
        response = self._response_queue.get()
        if response.get("type") == "result":
            return response["results"]
        raise RuntimeError(f"{self.config.name} generation failed:\n{response.get('error', response)}")

    def close(self, force: bool = False) -> None:
        if self._process.is_alive() and not force:
            try:
                self._request_queue.put({"type": "shutdown"})
                self._process.join(timeout=5)
            except Exception:
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout@N with one vLLM per GPU group.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _make_messages(question: str, template: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": template.replace("{question}", question)}]


def _chunks(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _build_worker_config(
    *,
    idx: int,
    gpus: str,
    rollout_cfg: Dict[str, Any],
) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        name=f"rollout-{idx}",
        gpus=gpus,
        model_path=str(rollout_cfg["model_path"]),
        tensor_parallel_size=int(rollout_cfg["tensor_parallel_size"]),
        dtype=str(rollout_cfg.get("dtype", "float16")),
        gpu_memory_utilization=float(rollout_cfg.get("gpu_memory_utilization", 0.9)),
        max_model_len=int(rollout_cfg["max_model_len"]),
        max_tokens=int(rollout_cfg["max_tokens"]),
        n=int(rollout_cfg["n"]),
        temperature=float(rollout_cfg["temperature"]),
        top_p=float(rollout_cfg["top_p"]),
        top_k=int(rollout_cfg["top_k"]),
        seed=int(rollout_cfg["seed"]) + idx,
        chat_template_kwargs=dict(rollout_cfg.get("chat_template_kwargs") or {}),
    )


def _start_workers(
    *,
    gpu_groups: List[str],
    rollout_cfg: Dict[str, Any],
) -> List[RolloutWorkerClient]:
    parallel_startup = bool(rollout_cfg.get("parallel_startup", True))
    startup_stagger_seconds = float(rollout_cfg.get("startup_stagger_seconds", 0) or 0)
    startup_timeout = int(rollout_cfg.get("startup_timeout", 1800))
    workers: List[RolloutWorkerClient] = []

    if parallel_startup:
        try:
            for idx, gpus in enumerate(gpu_groups):
                workers.append(
                    RolloutWorkerClient(
                        _build_worker_config(idx=idx, gpus=gpus, rollout_cfg=rollout_cfg),
                        startup_timeout=startup_timeout,
                        wait_ready=False,
                    )
                )
                print(f"[init] started rollout-{idx} on gpus={gpus}", flush=True)
                if startup_stagger_seconds > 0 and idx + 1 < len(gpu_groups):
                    time.sleep(startup_stagger_seconds)
            for idx, worker in enumerate(workers):
                worker.wait_until_ready(startup_timeout)
                print(f"[init] rollout-{idx} ready on gpus={worker.config.gpus}", flush=True)
            return workers
        except Exception:
            for worker in workers:
                worker.close(force=True)
            raise

    for idx, gpus in enumerate(gpu_groups):
        workers.append(
            RolloutWorkerClient(
                _build_worker_config(idx=idx, gpus=gpus, rollout_cfg=rollout_cfg),
                startup_timeout=startup_timeout,
                wait_ready=True,
            )
        )
        print(f"[init] rollout-{idx} ready on gpus={gpus}", flush=True)
    return workers


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rollout_cfg = cfg["rollout"]
    out_dir = exp_dir(cfg) / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = exp_dir(cfg) / "input" / "bench.parquet"
    output_path = out_dir / "rollout_raw.jsonl"

    df = pd.read_parquet(input_path)
    done_ids = read_done_ids(output_path) if rollout_cfg.get("resume", True) else set()
    if done_ids:
        df = df[~df["id"].astype(str).isin(done_ids)]
        print(f"[resume] skipping {len(done_ids)} completed prompt(s)")
    if df.empty:
        print("[done] no prompts to rollout")
        return

    template = cfg["prompt"]["template"]
    items = []
    for rec in df.to_dict("records"):
        items.append(
            {
                "id": str(rec["id"]),
                "source": str(rec["source"]),
                "question": str(rec["question"]),
                "gold": str(rec["gold"]),
                "messages": _make_messages(str(rec["question"]), template),
            }
        )

    gpu_groups = split_gpu_groups(
        str(rollout_cfg["gpus"]),
        instances=int(rollout_cfg["instances"]),
        tensor_parallel_size=int(rollout_cfg["tensor_parallel_size"]),
    )
    workers: List[RolloutWorkerClient] = []
    try:
        workers = _start_workers(gpu_groups=gpu_groups, rollout_cfg=rollout_cfg)
        print(f"[init] rollout workers ready: {gpu_groups}")

        micro_batch_size = int(rollout_cfg["micro_batch_size"])
        batches = _chunks(items, micro_batch_size)
        progress = ProgressBar(
            total=len(items),
            desc=f"rollout@{int(rollout_cfg['n'])}",
            enabled=bool(rollout_cfg.get("progress", True)),
        )
        mode = "a" if output_path.exists() and rollout_cfg.get("resume", True) else "w"
        try:
            with output_path.open(mode, encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(
                max_workers=len(workers)
            ) as executor:
                next_batch_idx = 0
                future_map: Dict[concurrent.futures.Future, int] = {}

                for worker_id, worker in enumerate(workers):
                    if next_batch_idx >= len(batches):
                        break
                    future = executor.submit(worker.generate, batches[next_batch_idx])
                    future_map[future] = worker_id
                    print(
                        f"[rollout:{worker_id}] dispatched batch={next_batch_idx} "
                        f"size={len(batches[next_batch_idx])}",
                        flush=True,
                    )
                    next_batch_idx += 1

                completed_batches = 0
                while future_map:
                    done, _ = concurrent.futures.wait(
                        future_map,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        worker_id = future_map.pop(future)
                        results = future.result()
                        for rec in results:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        completed_batches += 1
                        progress.update(len(results))
                        print(
                            f"[rollout:{worker_id}] wrote {len(results)} prompt(s) "
                            f"completed_batches={completed_batches}/{len(batches)}",
                            flush=True,
                        )

                        if next_batch_idx < len(batches):
                            future_next = executor.submit(workers[worker_id].generate, batches[next_batch_idx])
                            future_map[future_next] = worker_id
                            print(
                                f"[rollout:{worker_id}] dispatched batch={next_batch_idx} "
                                f"size={len(batches[next_batch_idx])}",
                                flush=True,
                            )
                            next_batch_idx += 1
        finally:
            progress.close()
        print(f"[done] rollout -> {output_path}")
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
