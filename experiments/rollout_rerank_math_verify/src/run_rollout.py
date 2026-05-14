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

from common import load_config, read_done_ids, rollout_dir
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
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--limit-total", type=int, default=None)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def _make_messages(question: str, template: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": template.replace("{question}", question)}]


def _chunks(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _load_bench(cfg: Dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    data_cfg = cfg["data"]
    bench_path = Path(data_cfg["bench_path"])
    if bench_path.suffix == ".parquet":
        df = pd.read_parquet(bench_path)
    elif bench_path.suffix == ".jsonl":
        df = pd.read_json(bench_path, lines=True)
    else:
        raise SystemExit(f"unsupported bench file: {bench_path}")

    columns = {
        "id": data_cfg.get("id_column", "id"),
        "source": data_cfg.get("source_column", "source"),
        "question": data_cfg.get("question_column", "question"),
        "gold": data_cfg.get("gold_column", "gold"),
    }
    missing = [col for col in columns.values() if col not in df.columns]
    if missing:
        raise SystemExit(f"missing required column(s): {missing}")

    df = df.rename(columns={v: k for k, v in columns.items()})
    df = df[["id", "source", "question", "gold"]].copy()
    df = df.dropna(subset=["id", "source", "question", "gold"])
    df["id"] = df["id"].astype(str)
    df["source"] = df["source"].astype(str)
    df["question"] = df["question"].astype(str)
    df["gold"] = df["gold"].astype(str)

    sources = data_cfg.get("sources")
    if sources:
        available_sources = set(df["source"].unique().tolist())
        requested_sources = set(str(source) for source in sources)
        unknown_sources = sorted(requested_sources - available_sources)
        if unknown_sources:
            available = ", ".join(sorted(available_sources))
            raise SystemExit(
                f"unknown data.sources value(s): {unknown_sources}. "
                f"Available sources: {available}"
            )
        df = df[df["source"].isin(sources)]
        if df.empty:
            raise SystemExit(f"data.sources selected no records: {sources}")

    seed = int(data_cfg.get("seed", 42))
    limit_per_source = (
        args.limit_per_source
        if args.limit_per_source is not None
        else data_cfg.get("limit_per_source")
    )
    if limit_per_source:
        parts = []
        for _, group in df.groupby("source", sort=True):
            parts.append(
                group.sample(
                    n=min(int(limit_per_source), len(group)),
                    random_state=seed,
                )
            )
        df = pd.concat(parts, ignore_index=True)

    if args.limit_total:
        df = df.sample(n=min(args.limit_total, len(df)), random_state=seed)

    return df.sort_values(["source", "id"]).reset_index(drop=True)


def _write_manifest(out_dir: Path, cfg: Dict[str, Any], df: pd.DataFrame) -> None:
    data_cfg = cfg["data"]
    manifest = {
        "name": cfg["name"],
        "bench_path": str(data_cfg["bench_path"]),
        "total": int(len(df)),
        "sources": {str(k): int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "data": {
            "id_column": data_cfg.get("id_column", "id"),
            "source_column": data_cfg.get("source_column", "source"),
            "question_column": data_cfg.get("question_column", "question"),
            "gold_column": data_cfg.get("gold_column", "gold"),
            "limit_per_source": data_cfg.get("limit_per_source"),
            "sources": data_cfg.get("sources"),
            "seed": data_cfg.get("seed", 42),
        },
        "rollout": {
            "model_path": cfg["rollout"].get("model_path"),
            "n": cfg["rollout"].get("n"),
            "temperature": cfg["rollout"].get("temperature"),
            "top_p": cfg["rollout"].get("top_p"),
            "top_k": cfg["rollout"].get("top_k"),
            "seed": cfg["rollout"].get("seed"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    cfg = load_config(args.config, args.overrides)
    rollout_cfg = cfg["rollout"]
    out_dir = rollout_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "rollout_raw.jsonl"

    df = _load_bench(cfg, args)
    _write_manifest(out_dir, cfg, df)
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
