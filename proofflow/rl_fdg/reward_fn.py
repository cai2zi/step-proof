from __future__ import annotations

import atexit
import json
from typing import Any, Dict, List, Optional

from .evaluator import FDGRLEvaluator, load_evaluator_config
from .reward_types import CandidateGraphInput


_RUNTIME: Optional[FDGRLEvaluator] = None
_RUNTIME_KEY: Optional[str] = None


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


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    reward_config_path: str,
    **reward_kwargs: Any,
) -> List[Dict[str, Any]]:
    runtime = _get_runtime(reward_config_path=reward_config_path, **reward_kwargs)
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
    return [breakdown.to_reward_dict() for breakdown in runtime.evaluate_batch_sync(batch_inputs)]
