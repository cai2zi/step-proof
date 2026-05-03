from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# verl 通过文件路径动态加载本模块时 __package__ 为空，相对导入会失败；需保证可导入 proofflow.*
_repo_root = Path(__file__).resolve().parents[2]
_root_s = str(_repo_root)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

from proofflow.rl_fdg.evaluator import FDGRLEvaluator, load_evaluator_config
from proofflow.rl_fdg.reward_types import CandidateGraphInput


_RUNTIME: Optional[FDGRLEvaluator] = None
_RUNTIME_KEY: Optional[str] = None
_RUNTIME_LOCK = threading.RLock()


def _runtime_key(config_path: str, extra_kwargs: Dict[str, Any]) -> str:
    return json.dumps({"config_path": str(config_path), "extra_kwargs": extra_kwargs}, sort_keys=True)


def _close_runtime() -> None:
    global _RUNTIME, _RUNTIME_KEY
    if _RUNTIME is None:
        return
    try:
        _RUNTIME.close()
    finally:
        _RUNTIME = None
        _RUNTIME_KEY = None


atexit.register(_close_runtime)


def _get_runtime(*, reward_config_path: str, **reward_kwargs: Any) -> FDGRLEvaluator:
    global _RUNTIME, _RUNTIME_KEY
    key = _runtime_key(reward_config_path, reward_kwargs)
    if _RUNTIME is not None and _RUNTIME_KEY == key:
        return _RUNTIME
    _close_runtime()
    config = load_evaluator_config(reward_config_path)
    _RUNTIME = FDGRLEvaluator(config)
    _RUNTIME_KEY = key
    return _RUNTIME


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _compute_local(
    batch_inputs: List[CandidateGraphInput],
    *,
    reward_config_path: str,
    reward_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # FDGRLEvaluator 复用一个 asyncio loop 和 vLLM/Lean runtime，不能被多个线程同时 run_until_complete。
    with _RUNTIME_LOCK:
        runtime = _get_runtime(reward_config_path=reward_config_path, **reward_kwargs)
        return [breakdown.to_reward_dict() for breakdown in runtime.evaluate_batch_sync(batch_inputs)]


def _compute_with_actor(
    batch_inputs: List[CandidateGraphInput],
    *,
    reward_config_path: str,
    runtime_actor_name: str,
    runtime_actor_namespace: str,
) -> List[Dict[str, Any]]:
    import ray

    from proofflow.rl_fdg.runtime_actor import get_or_create_runtime_actor

    actor = get_or_create_runtime_actor(
        reward_config_path=reward_config_path,
        runtime_actor_name=runtime_actor_name,
        runtime_actor_namespace=runtime_actor_namespace,
    )
    return ray.get(actor.evaluate.remote([asdict(item) for item in batch_inputs]))


def compute_score(
    data_sources: Optional[List[str]] = None,
    solution_strs: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    extra_infos: Optional[List[Dict[str, Any]]] = None,
    reward_config_path: str = "",
    use_runtime_actor: bool | str = True,
    runtime_actor_name: str = "fdg_rl_runtime",
    runtime_actor_namespace: str = "step_proof_rl",
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **reward_kwargs: Any,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    if not reward_config_path:
        raise ValueError("reward_config_path is required")

    single_item = data_sources is None
    if single_item:
        data_sources = [str(data_source or "")]
        solution_strs = [str(solution_str or "")]
        ground_truths = [str(ground_truth or "")]
        extra_infos = [dict(extra_info or {})]

    assert data_sources is not None
    assert solution_strs is not None
    assert ground_truths is not None
    assert extra_infos is not None

    batch_inputs: List[CandidateGraphInput] = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
    ):
        batch_inputs.append(
            CandidateGraphInput(
                record_id=str(extra_info.get("record_id", "")),
                problem_text=str(extra_info.get("problem_text", "")),
                solution_or_cot=str(extra_info.get("solution_or_cot", "")),
                model_output=solution_str,
                ground_truth=str(ground_truth or ""),
                data_source=str(data_source),
                extra_info=dict(extra_info),
            )
        )

    local_reward_kwargs = dict(reward_kwargs)
    rewards: List[Dict[str, Any]]
    if _coerce_bool(use_runtime_actor, default=True):
        try:
            rewards = _compute_with_actor(
                batch_inputs,
                reward_config_path=reward_config_path,
                runtime_actor_name=str(runtime_actor_name or "fdg_rl_runtime"),
                runtime_actor_namespace=str(runtime_actor_namespace or "step_proof_rl"),
            )
        except Exception as exc:
            if os.getenv("STEP_PROOF_RL_TRACE", "1") != "0":
                print(f"[rl_fdg] shared runtime actor unavailable, falling back to local evaluator: {exc}", flush=True)
            rewards = _compute_local(
                batch_inputs,
                reward_config_path=reward_config_path,
                reward_kwargs=local_reward_kwargs,
            )
    else:
        rewards = _compute_local(
            batch_inputs,
            reward_config_path=reward_config_path,
            reward_kwargs=local_reward_kwargs,
        )
    return rewards[0] if single_item else rewards
