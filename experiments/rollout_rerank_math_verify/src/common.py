from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - fallback for lightweight eval envs.
    OmegaConf = None  # type: ignore[assignment]


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
STEP_PROOF_ROOT = EXPERIMENT_DIR.parents[1]


def _set_dot_path(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = [part for part in key.split(".") if part]
    if not parts:
        return
    cur = cfg
    for part in parts[:-1]:
        next_value = cur.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[part] = next_value
        cur = next_value
    cur[parts[-1]] = value


def _select_dot_path(cfg: Dict[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _resolve_fallback(value: Any, root: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_fallback(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_fallback(item, root) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        if expr.startswith("oc.env:"):
            return match.group(0)
        try:
            return str(_select_dot_path(root, expr))
        except KeyError:
            return match.group(0)

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _load_config_fallback(path: str | Path, overrides: List[str] | None) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Config override must be key=value: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dot_path(cfg, key, yaml.safe_load(raw_value))
    return _resolve_fallback(cfg, cfg)


def load_config(path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    if OmegaConf is None:
        return _load_config_fallback(path, overrides)
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


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


def has_unclosed_think_block(text: Any) -> bool:
    value = "" if text is None else str(text)
    without_closed = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return bool(
        re.search(
            r"<think\b[^>]*>.*\Z",
            without_closed,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )


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
