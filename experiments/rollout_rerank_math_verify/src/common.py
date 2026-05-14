from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
STEP_PROOF_ROOT = EXPERIMENT_DIR.parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def exp_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["output_root"])


def _named_dir(prefix: str, name: str) -> str:
    return name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"


def rollout_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("rollout_name") or cfg["name"])


def rollout_dir(cfg: Dict[str, Any]) -> Path:
    return exp_dir(cfg) / "rollouts" / _named_dir("rollout", rollout_name(cfg))


def step_proof_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("step_proof_name") or cfg["name"])


def step_proof_rollout_dir(cfg: Dict[str, Any]) -> Path:
    return exp_dir(cfg) / "rollouts" / _named_dir("rollout", rollout_name(cfg))


def step_proof_dir(cfg: Dict[str, Any]) -> Path:
    return exp_dir(cfg) / "step_proofs" / _named_dir("step_proof", step_proof_name(cfg))


def step_proof_results_dir(cfg: Dict[str, Any]) -> Path:
    return step_proof_dir(cfg) / "step_proof_results"


def math_verify_dir(cfg: Dict[str, Any]) -> Path:
    return step_proof_dir(cfg) / "math_verify"


def summary_dir(cfg: Dict[str, Any]) -> Path:
    return step_proof_dir(cfg) / "summary"


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_done_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.is_file():
        return set()
    done = set()
    for rec in read_jsonl(p):
        rid = rec.get("id")
        if rid:
            done.add(str(rid))
    return done


def rollout_response_key(rollout_id: int) -> str:
    return f"response_{rollout_id}"


def rollout_record_id(parent_id: str, rollout_id: int) -> str:
    return f"{parent_id}__rollout_{rollout_id}"


def split_rollout_id(record_id: str) -> tuple[str, int]:
    marker = "__rollout_"
    if marker not in record_id:
        raise ValueError(f"not a rollout record id: {record_id}")
    parent_id, rollout = record_id.rsplit(marker, 1)
    return parent_id, int(rollout)


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
