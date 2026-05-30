from __future__ import annotations

import os
from pathlib import Path


def czx_root() -> Path:
    return Path(os.environ.get("CZX_ROOT", r"D:\program\research")).expanduser()


def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", czx_root() / "step-proof")).expanduser()


def output_root() -> Path:
    return Path(
        os.environ.get(
            "OUTPUT_ROOT",
            czx_root() / "czx_work" / "step-proof" / "rollout_rerank_math_verify" / "outputs",
        )
    ).expanduser()


def rollouts_root() -> Path:
    return Path(os.environ.get("ROLLOUT_ROOT", output_root() / "rollouts")).expanduser()


def step_proofs_root() -> Path:
    return Path(os.environ.get("STEP_PROOF_OUTPUT_ROOT", output_root() / "step_proofs")).expanduser()


def default_rollout_flat(name: str = "qwen3_8b") -> Path:
    return rollouts_root() / f"rollout_{name}" / "rollout_flat.parquet"

