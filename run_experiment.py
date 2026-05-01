from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hydra  # type: ignore[import-not-found]
from hydra.core.hydra_config import HydraConfig  # type: ignore[import-not-found]
from omegaconf import DictConfig, OmegaConf  # type: ignore[import-not-found]

from proofflow.experiment_lean_runtime import ExperimentLeanRuntime, LeanRuntimeConfig
from proofflow.fdg_stage2_runner import FDGStage2Runner, build_arg_parser as build_stage2_arg_parser
from proofflow.fdg_stage3_runner import FDGStage3Runner, build_arg_parser as build_stage3_arg_parser
from proofflow.graph_mode import ensure_fdg_jsonl


JsonDict = Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _as_path(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _exp_dir(cfg: DictConfig, repo_root: Path) -> Path:
    return _as_path(repo_root, str(cfg.exp.root)) / str(cfg.exp.name)


def _json_arg(value: Any) -> Optional[str]:
    if value is None:
        return None
    plain = OmegaConf.to_container(value, resolve=True)
    if plain is None:
        return None
    return json.dumps(plain, ensure_ascii=False)


def _bool_flag(enabled: bool, name: str) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _cmd_value(value: Any) -> str:
    return str(OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value)


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _stage1_done_count(graphs_jsonl: Path) -> int:
    return _count_jsonl(graphs_jsonl)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


class _TeeStream(io.TextIOBase):
    def __init__(self, log_f: io.TextIOBase, console_f: Optional[io.TextIOBase]) -> None:
        self.log_f = log_f
        self.console_f = console_f

    def write(self, s: str) -> int:
        self.log_f.write(s)
        if self.console_f is not None:
            self.console_f.write(s)
        return len(s)

    def flush(self) -> None:
        self.log_f.flush()
        if self.console_f is not None:
            self.console_f.flush()


class ExperimentRunner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.repo_root = _repo_root()
        self.exp_dir = _exp_dir(cfg, self.repo_root)
        self.logs_dir = self.exp_dir / "logs"
        self.stats_dir = self.exp_dir / "stats"
        self.status_path = self.stats_dir / "status.json"
        self.status: JsonDict = {}
        self.python = str(cfg.python)

        self.stage1_dir = self.exp_dir / "result_stage1"
        self.stage2_dir = self.exp_dir / "result_stage2"
        self.stage3_dir = self.exp_dir / "result_stage3"
        self.cot_dir = self.exp_dir / "cot_traces"
        self.viz_dir = self.exp_dir / "visualizations"
        self.started_at = _utc_now_iso()
        self.runtime_metrics_path = self.stats_dir / "stage_runtime_metrics.json"
        self.shared_lean_temp_dir = (
            self._shared_lean_temp_dir() if "lean_runtime" in self.cfg else self.exp_dir / "lean_jobs_shared"
        )

    def _shared_lean_cfg(self) -> DictConfig:
        if "lean_runtime" not in self.cfg:
            raise RuntimeError("Missing required top-level config block: lean_runtime")
        return self.cfg.lean_runtime

    def _shared_lean_temp_dir(self) -> Path:
        cfg = self._shared_lean_cfg()
        if "lean_temp_dir" not in cfg:
            raise RuntimeError("lean_runtime.lean_temp_dir is required")
        return _as_path(self.exp_dir, _cmd_value(cfg.lean_temp_dir))

    def _build_shared_lean_runtime_config(self) -> LeanRuntimeConfig:
        cfg = self._shared_lean_cfg()
        required = [
            "mathlib_path",
            "lean_backend",
            "lean_check_concurrency",
            "lean_worker_pool_size",
            "lean_temp_dir",
        ]
        missing = [name for name in required if name not in cfg]
        if missing:
            raise RuntimeError(f"lean_runtime is missing required fields: {missing}")
        mathlib_path = _cmd_value(cfg.mathlib_path)
        if not Path(mathlib_path).is_dir():
            raise RuntimeError(f"lean_runtime.mathlib_path is not a directory: {mathlib_path}")
        return LeanRuntimeConfig(
            mathlib_path=mathlib_path,
            lean_backend=_cmd_value(cfg.lean_backend),
            lean_check_concurrency=int(cfg.lean_check_concurrency),
            lean_worker_pool_size=int(cfg.lean_worker_pool_size),
            lean_temp_dir=str(self.shared_lean_temp_dir),
        )

    def prepare(self) -> None:
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        if bool(self.cfg.run.force):
            for path in (
                self.logs_dir,
                self.stage1_dir,
                self.stage2_dir,
                self.stage3_dir,
                self.stats_dir,
                self.cot_dir,
                self.viz_dir,
                self.shared_lean_temp_dir,
            ):
                if path.exists():
                    shutil.rmtree(path)
        for path in (
            self.logs_dir,
            self.stage1_dir,
            self.stage2_dir,
            self.stage3_dir,
            self.stats_dir,
            self.cot_dir,
            self.viz_dir,
            self.shared_lean_temp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._write_resolved_config()
        self._write_run_meta(started=True)

    def _write_resolved_config(self) -> None:
        (self.stats_dir / "config_resolved.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True),
            encoding="utf-8",
        )

    def _write_run_meta(self, *, started: bool = False, ended: bool = False) -> None:
        hydra_cfg = HydraConfig.get()
        payload = {
            "exp_name": str(self.cfg.exp.name),
            "exp_dir": str(self.exp_dir),
            "repo_root": str(self.repo_root),
            "git_commit": _git_commit(self.repo_root),
            "python": sys.version,
            "python_executable": sys.executable,
            "runner_python": self.python,
            "hydra_overrides": list(hydra_cfg.overrides.task),
        }
        payload["started_at"] = self.started_at
        if ended:
            payload["ended_at"] = _utc_now_iso()
        _write_json(self.stats_dir / "run_meta.json", payload)

    def _update_status(self, stage: str, payload: JsonDict) -> None:
        self.status[stage] = payload
        _write_json(self.status_path, self.status)

    def _load_runtime_metrics(self) -> JsonDict:
        if not self.runtime_metrics_path.is_file():
            return {"stages": {}, "run": {}}
        return json.loads(self.runtime_metrics_path.read_text(encoding="utf-8"))

    def _update_runtime_metrics(self, stage: str, payload: JsonDict) -> None:
        metrics = self._load_runtime_metrics()
        metrics.setdefault("stages", {})
        metrics["stages"][stage] = payload
        _write_json(self.runtime_metrics_path, metrics)

    def _merge_stage_detail_metrics(self, stage: str) -> None:
        detail_path_map = {
            "stage2": self.stats_dir / "stage2_runtime_stats.json",
            "stage3": self.stats_dir / "stage3_runtime_stats.json",
        }
        detail_path = detail_path_map.get(stage)
        if detail_path is None or not detail_path.is_file():
            return
        detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
        metrics = self._load_runtime_metrics()
        metrics.setdefault("stages", {})
        stage_payload = metrics["stages"].get(stage, {})
        if isinstance(stage_payload, dict):
            stage_payload.update(detail_payload)
            metrics["stages"][stage] = stage_payload
            _write_json(self.runtime_metrics_path, metrics)

    def _finalize_runtime_metrics(self) -> None:
        metrics = self._load_runtime_metrics()
        run_started = self.started_at
        run_ended = _utc_now_iso()
        duration_sum = 0.0
        for stage_payload in metrics.get("stages", {}).values():
            duration_sum += float(stage_payload.get("duration_seconds", 0.0) or 0.0)
        metrics["run"] = {
            "started_at": run_started,
            "ended_at": run_ended,
            "total_stage_duration_seconds": round(duration_sum, 6),
        }
        _write_json(self.runtime_metrics_path, metrics)

    def _finalize_stage_status(
        self,
        stage: str,
        cmd: List[str],
        log_path: Path,
        started_at: str,
        started_perf: float,
        result_code: int,
    ) -> None:
        ended_at = _utc_now_iso()
        duration_seconds = time.perf_counter() - started_perf
        status = {
            "command": cmd,
            "log_path": str(log_path),
            "started_at": started_at,
            "ended_at": ended_at,
            "exit_code": result_code,
            "duration_seconds": round(duration_seconds, 6),
        }
        self._update_status(stage, status)
        self._update_runtime_metrics(
            stage,
            {
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": round(duration_seconds, 6),
                "exit_code": result_code,
                "log_path": str(log_path),
                "command": cmd,
            },
        )
        self._merge_stage_detail_metrics(stage)

    def _run_command(self, stage: str, cmd: List[str]) -> None:
        log_path = self.logs_dir / f"{stage}.log"
        started_at = _utc_now_iso()
        started_perf = time.perf_counter()
        stream_to_console = bool(self.cfg.run.stream_logs_to_console)
        self._update_status(
            stage,
            {
                "command": cmd,
                "log_path": str(log_path),
                "started_at": started_at,
                "exit_code": None,
            },
        )
        with open(log_path, "w", encoding="utf-8") as log_f:
            cmd_line = "$ " + " ".join(cmd)
            log_f.write(cmd_line + "\n\n")
            log_f.flush()
            if stream_to_console:
                print(f"[{stage}] {cmd_line}", flush=True)
            process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_f.write(line)
                if stream_to_console:
                    print(line, end="", flush=True)
            process.wait()
            result_code = process.returncode
        self._finalize_stage_status(stage, cmd, log_path, started_at, started_perf, result_code)
        if result_code != 0:
            raise RuntimeError(f"{stage} failed with exit code {result_code}; see {log_path}")

    async def _run_inprocess_stage(
        self,
        stage: str,
        cmd: List[str],
        runner_coro: Callable[[], asyncio.Future[Any] | Any],
    ) -> None:
        log_path = self.logs_dir / f"{stage}.log"
        started_at = _utc_now_iso()
        started_perf = time.perf_counter()
        stream_to_console = bool(self.cfg.run.stream_logs_to_console)
        self._update_status(
            stage,
            {
                "command": cmd,
                "log_path": str(log_path),
                "started_at": started_at,
                "exit_code": None,
            },
        )
        result_code = 0
        with open(log_path, "w", encoding="utf-8") as log_f:
            cmd_line = "$ " + " ".join(cmd)
            log_f.write(cmd_line + "\n\n")
            log_f.flush()
            if stream_to_console:
                print(f"[{stage}] {cmd_line}", flush=True)
            tee = _TeeStream(log_f, sys.__stdout__ if stream_to_console else None)
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    await runner_coro()
            except Exception:
                result_code = 1
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    traceback.print_exc()
                raise
            finally:
                tee.flush()
                self._finalize_stage_status(stage, cmd, log_path, started_at, started_perf, result_code)

    def _build_stage1_cmd(self, effective_limit: int) -> List[str]:
        cfg = self.cfg.stage1
        cmd = [
            self.python,
            "-u",
            str(self.repo_root / "build_calc_graph_stage1.py"),
            "--parquet-dir",
            _cmd_value(cfg.parquet_dir),
            "--glob",
            _cmd_value(cfg.parquet_glob),
            "--id-column",
            _cmd_value(cfg.id_column),
            "--question-column",
            _cmd_value(cfg.question_column),
            "--response-column",
            _cmd_value(cfg.response_column),
            "--limit",
            str(effective_limit),
            "--out",
            str(self.stage1_dir / "graphs.jsonl"),
            "--skipped",
            str(self.stage1_dir / "skipped.jsonl"),
            "--failed",
            str(self.stage1_dir / "failed.jsonl"),
            "--model-path",
            _cmd_value(cfg.model_path),
            "--tensor-parallel-size",
            _cmd_value(cfg.tensor_parallel_size),
            "--gpus",
            _cmd_value(cfg.gpus),
            "--dtype",
            _cmd_value(cfg.dtype),
            "--gpu-memory-utilization",
            _cmd_value(cfg.gpu_memory_utilization),
            "--max-tokens",
            _cmd_value(cfg.max_tokens),
            "--temperature",
            _cmd_value(cfg.temperature),
            "--top-p",
            _cmd_value(cfg.top_p),
            "--presence-penalty",
            _cmd_value(cfg.presence_penalty),
            "--frequency-penalty",
            _cmd_value(cfg.frequency_penalty),
            "--seed",
            _cmd_value(cfg.seed),
            "--top-k",
            _cmd_value(cfg.top_k),
            "--token-limit",
            _cmd_value(cfg.token_limit),
            "--batch-size",
            _cmd_value(cfg.batch_size),
            "--max-retries",
            _cmd_value(cfg.max_retries),
            "--fdg-prompt",
            _cmd_value(cfg.fdg_prompt),
            _bool_flag(bool(cfg.include_think_in_dag), "include-think-in-dag"),
        ]
        chat_kwargs = _json_arg(cfg.chat_template_kwargs)
        if chat_kwargs:
            cmd.extend(["--chat-template-kwargs-json", chat_kwargs])
        if not bool(self.cfg.run.resume):
            cmd.append("--no-resume")
        return cmd

    def _build_stage2_cmd(self) -> List[str]:
        cfg = self.cfg.stage2
        lean_cfg = self._shared_lean_cfg()
        cmd = [
            self.python,
            "-u",
            str(self.repo_root / "build_calc_graph_stage2.py"),
            "--infile",
            str(self.stage1_dir / "graphs.jsonl"),
            "--out",
            str(self.stage2_dir / "stage2_results.jsonl"),
            "--failed",
            str(self.stage2_dir / "stage2_failed.jsonl"),
            "--checkpoint-dir",
            str(self.stage2_dir / "stage2_ckpt"),
            "--limit",
            _cmd_value(cfg.limit),
            "--mathlib-path",
            _cmd_value(lean_cfg.mathlib_path),
            "--lean-backend",
            _cmd_value(lean_cfg.lean_backend),
            "--lean-check-concurrency",
            _cmd_value(lean_cfg.lean_check_concurrency),
            "--lean-worker-pool-size",
            _cmd_value(lean_cfg.lean_worker_pool_size),
            "--lean-temp-dir",
            str(self.shared_lean_temp_dir),
            "--gpus",
            _cmd_value(cfg.gpus),
            "--dtype",
            _cmd_value(cfg.dtype),
            "--gpu-memory-utilization",
            _cmd_value(cfg.gpu_memory_utilization),
            "--batch-wait-ms",
            _cmd_value(cfg.batch_wait_ms),
            "--max-pending-validation-batches",
            _cmd_value(cfg.max_pending_validation_batches),
            "--formalizer-model-path",
            _cmd_value(cfg.formalizer_model_path),
            "--formalizer-tensor-parallel-size",
            _cmd_value(cfg.formalizer_tensor_parallel_size),
            "--formalizer-max-tokens",
            _cmd_value(cfg.formalizer_max_tokens),
            "--formalizer-token-limit",
            _cmd_value(cfg.formalizer_token_limit),
            "--formalizer-temperature",
            _cmd_value(cfg.formalizer_temperature),
            "--formalizer-top-p",
            _cmd_value(cfg.formalizer_top_p),
            "--formalizer-presence-penalty",
            _cmd_value(cfg.formalizer_presence_penalty),
            "--formalizer-frequency-penalty",
            _cmd_value(cfg.formalizer_frequency_penalty),
            "--formalizer-seed",
            _cmd_value(cfg.formalizer_seed),
            "--formalizer-top-k",
            _cmd_value(cfg.formalizer_top_k),
            "--formalizer-retries",
            _cmd_value(cfg.formalizer_retries),
            "--formalizer-prompt",
            _cmd_value(cfg.formalizer_prompt),
            "--form-batch-size",
            _cmd_value(cfg.form_batch_size),
            "--metrics-out",
            str(self.stats_dir / "stage2_runtime_stats.json"),
        ]
        chat_kwargs = _json_arg(cfg.formalizer_chat_template_kwargs)
        if chat_kwargs:
            cmd.extend(["--formalizer-chat-template-kwargs-json", chat_kwargs])
        if not bool(self.cfg.run.resume):
            cmd.append("--no-resume")
        return cmd

    def _build_stage3_cmd(self) -> List[str]:
        cfg = self.cfg.stage3
        lean_cfg = self._shared_lean_cfg()
        cmd = [
            self.python,
            "-u",
            str(self.repo_root / "build_calc_graph_stage3.py"),
            "--infile",
            str(self.stage2_dir / "stage2_results.jsonl"),
            "--out",
            str(self.stage3_dir / "stage3_results.jsonl"),
            "--failed",
            str(self.stage3_dir / "stage3_failed.jsonl"),
            "--checkpoint-dir",
            str(self.stage3_dir / "stage3_ckpt"),
            "--limit",
            _cmd_value(cfg.limit),
            "--mathlib-path",
            _cmd_value(lean_cfg.mathlib_path),
            "--lean-backend",
            _cmd_value(lean_cfg.lean_backend),
            "--lean-check-concurrency",
            _cmd_value(lean_cfg.lean_check_concurrency),
            "--lean-worker-pool-size",
            _cmd_value(lean_cfg.lean_worker_pool_size),
            "--lean-temp-dir",
            str(self.shared_lean_temp_dir),
            "--gpus",
            _cmd_value(cfg.gpus),
            "--dtype",
            _cmd_value(cfg.dtype),
            "--gpu-memory-utilization",
            _cmd_value(cfg.gpu_memory_utilization),
            "--batch-wait-ms",
            _cmd_value(cfg.batch_wait_ms),
            "--max-pending-validation-batches",
            _cmd_value(cfg.max_pending_validation_batches),
            "--prover-model-path",
            _cmd_value(cfg.prover_model_path),
            "--prover-tensor-parallel-size",
            _cmd_value(cfg.prover_tensor_parallel_size),
            "--prover-max-tokens",
            _cmd_value(cfg.prover_max_tokens),
            "--prover-token-limit",
            _cmd_value(cfg.prover_token_limit),
            "--prover-temperature",
            _cmd_value(cfg.prover_temperature),
            "--prover-top-p",
            _cmd_value(cfg.prover_top_p),
            "--prover-presence-penalty",
            _cmd_value(cfg.prover_presence_penalty),
            "--prover-frequency-penalty",
            _cmd_value(cfg.prover_frequency_penalty),
            "--prover-seed",
            _cmd_value(cfg.prover_seed),
            "--prover-top-k",
            _cmd_value(cfg.prover_top_k),
            "--prover-retries",
            _cmd_value(cfg.prover_retries),
            "--prover-prompt",
            _cmd_value(cfg.prover_prompt),
            "--prove-batch-size",
            _cmd_value(cfg.prove_batch_size),
            "--metrics-out",
            str(self.stats_dir / "stage3_runtime_stats.json"),
        ]
        chat_kwargs = _json_arg(cfg.prover_chat_template_kwargs)
        if chat_kwargs:
            cmd.extend(["--prover-chat-template-kwargs-json", chat_kwargs])
        if not bool(self.cfg.run.resume):
            cmd.append("--no-resume")
        return cmd

    def _build_stage2_runner(self, args: argparse.Namespace, runtime: ExperimentLeanRuntime):
        if not args.infile.is_file():
            raise RuntimeError(f"--infile not found: {args.infile}")
        if not Path(args.mathlib_path).is_dir():
            raise RuntimeError(f"--mathlib-path is not a directory: {args.mathlib_path}")
        ensure_fdg_jsonl(args.infile)
        return FDGStage2Runner(args, lean_server=runtime.lean_server, owned_lean_server=False)

    def _build_stage3_runner(self, args: argparse.Namespace, runtime: ExperimentLeanRuntime):
        if not args.infile.is_file():
            raise RuntimeError(f"--infile not found: {args.infile}")
        if not Path(args.mathlib_path).is_dir():
            raise RuntimeError(f"--mathlib-path is not a directory: {args.mathlib_path}")
        ensure_fdg_jsonl(args.infile)
        return FDGStage3Runner(args, lean_server=runtime.lean_server, owned_lean_server=False)

    async def _run_stage2_stage3_with_runtime(
        self,
        runtime: ExperimentLeanRuntime,
        *,
        run_stage2: bool,
        run_stage3: bool,
    ) -> None:
        await runtime.ensure_ready()
        if run_stage2:
            stage2_cmd = self._build_stage2_cmd()
            stage2_args = build_stage2_arg_parser().parse_args(stage2_cmd[3:])
            runner = self._build_stage2_runner(stage2_args, runtime)
            await self._run_inprocess_stage("stage2", stage2_cmd, runner.run)
        if run_stage3:
            stage3_cmd = self._build_stage3_cmd()
            stage3_args = build_stage3_arg_parser().parse_args(stage3_cmd[3:])
            runner = self._build_stage3_runner(stage3_args, runtime)
            await self._run_inprocess_stage("stage3", stage3_cmd, runner.run)

    async def _run_shared_lean_stages(self, *, run_stage2: bool, run_stage3: bool) -> None:
        runtime = ExperimentLeanRuntime(self._build_shared_lean_runtime_config())
        try:
            await self._run_stage2_stage3_with_runtime(
                runtime,
                run_stage2=run_stage2,
                run_stage3=run_stage3,
            )
        finally:
            await runtime.aclose()

    async def _run_stage1_then_shared_lean_stages(
        self,
        *,
        run_stage2: bool,
        run_stage3: bool,
    ) -> None:
        runtime = ExperimentLeanRuntime(self._build_shared_lean_runtime_config())
        prewarm_task = asyncio.create_task(runtime.ensure_ready())
        try:
            print("[lean-runtime] prewarming in parallel with stage1.\n", flush=True)
            await asyncio.to_thread(self.run_stage1)
            await prewarm_task
            await self._run_stage2_stage3_with_runtime(
                runtime,
                run_stage2=run_stage2,
                run_stage3=run_stage3,
            )
        except Exception:
            if not prewarm_task.done():
                prewarm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prewarm_task
            raise
        finally:
            await runtime.aclose()

    def run(self) -> None:
        stages = set(str(stage) for stage in self.cfg.run.stages)
        has_stage1 = "stage1" in stages
        has_shared_lean_stages = "stage2" in stages or "stage3" in stages
        if has_stage1 and has_shared_lean_stages:
            asyncio.run(
                self._run_stage1_then_shared_lean_stages(
                    run_stage2="stage2" in stages,
                    run_stage3="stage3" in stages,
                )
            )
        else:
            if has_stage1:
                self.run_stage1()
            if has_shared_lean_stages:
                asyncio.run(
                    self._run_shared_lean_stages(
                        run_stage2="stage2" in stages,
                        run_stage3="stage3" in stages,
                    )
                )
        if "stats" in stages:
            self.run_stats()
        if "cot" in stages:
            self.run_cot()
        if "viz" in stages:
            self.run_viz()
        self._write_run_meta(ended=True)
        self._finalize_runtime_metrics()

    def run_stage1(self) -> None:
        cfg = self.cfg.stage1
        graphs_out = self.stage1_dir / "graphs.jsonl"
        requested_limit = int(cfg.limit)
        effective_limit = requested_limit
        existing_done = 0
        if requested_limit >= 0 and bool(self.cfg.run.resume):
            existing_done = _stage1_done_count(graphs_out)
            effective_limit = max(requested_limit - existing_done, 0)

        print(
            f"[stage1] limit_requested={requested_limit} "
            f"existing_done={existing_done} effective_new_limit={effective_limit}",
            flush=True,
        )
        self._run_command("stage1", self._build_stage1_cmd(effective_limit))

    def run_stats(self) -> None:
        cmd = [
            self.python,
            "-u",
            str(self.repo_root / "check_stage3_fully_verified.py"),
            "--stage3-jsonl",
            str(self.stage3_dir / "stage3_results.jsonl"),
            "--out-json",
            str(self.stats_dir / "stage3_verify_stats.json"),
            "--top-n-per-bucket",
            _cmd_value(self.cfg.stats.top_n_per_bucket),
        ]
        if bool(self.cfg.stats.show_ids):
            cmd.append("--show-ids")
        self._run_command("stats", cmd)
        self.write_failure_summary()

    def write_failure_summary(self) -> None:
        summary = {
            "stage1": {
                "graphs": _count_jsonl(self.stage1_dir / "graphs.jsonl"),
                "skipped": _count_jsonl(self.stage1_dir / "skipped.jsonl"),
                "failed": _count_jsonl(self.stage1_dir / "failed.jsonl"),
            },
            "stage2": {
                "results": _count_jsonl(self.stage2_dir / "stage2_results.jsonl"),
                "failed": _count_jsonl(self.stage2_dir / "stage2_failed.jsonl"),
            },
            "stage3": {
                "results": _count_jsonl(self.stage3_dir / "stage3_results.jsonl"),
                "failed": _count_jsonl(self.stage3_dir / "stage3_failed.jsonl"),
            },
        }
        _write_json(self.stats_dir / "failure_summary.json", summary)

    def run_cot(self) -> None:
        cmd = [
            self.python,
            "-u",
            str(self.repo_root / "collect_cot_traces.py"),
            "--stage2-jsonl",
            str(self.stage2_dir / "stage2_results.jsonl"),
            "--stage3-jsonl",
            str(self.stage3_dir / "stage3_results.jsonl"),
            "--out-dir",
            str(self.cot_dir),
            _bool_flag(bool(self.cfg.cot_trace.include_attempt_history), "include-attempt-history"),
        ]
        self._run_command("cot", cmd)

    def _load_stats_bucket_ids(self, bucket: str, top_n: int) -> List[str]:
        stats_path = self.stats_dir / "stage3_verify_stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        ids_by_bucket = payload.get("prove_verify_ratio_distribution_top_ids") or payload.get(
            "prove_verify_ratio_distribution_top5_ids", {}
        )
        ids = ids_by_bucket.get(bucket, [])
        return [str(rid) for rid in ids[:top_n]]

    def run_viz(self) -> None:
        if not bool(self.cfg.viz.enabled):
            return
        for bucket in self.cfg.viz.buckets:
            bucket_label = str(bucket)
            record_ids = self._load_stats_bucket_ids(bucket_label, int(self.cfg.viz.top_n))
            if not record_ids:
                continue
            out_dir = self.viz_dir / bucket_label
            cmd = [
                self.python,
                "-u",
                str(self.repo_root / "visualize_calc_graph_stage2.py"),
                "--stage2-jsonl",
                str(self.stage3_dir / "stage3_results.jsonl"),
                "--source",
                _cmd_value(self.cfg.viz.source),
                "--seed",
                _cmd_value(self.cfg.viz.seed),
                "--out-dir",
                str(out_dir),
                "--record-ids",
                ",".join(record_ids),
            ]
            if bool(self.cfg.viz.graph_only):
                cmd.append("--graph-only")
            self._run_command(f"viz_{bucket_label}", cmd)


@hydra.main(version_base=None, config_path="configs", config_name="experiment_fdg")
def main(cfg: DictConfig) -> None:
    runner = ExperimentRunner(cfg)
    runner.prepare()
    runner.run()


if __name__ == "__main__":
    main()
