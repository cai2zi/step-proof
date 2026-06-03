from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    """Normalized model/runtime knobs shared by API and vLLM backends."""

    backend: str
    name: str
    model_path: str = ""
    api_base_url: str = ""
    api_key_env: str = ""
    instances: int = 1
    tensor_parallel_size: int = 1
    gpus: str = ""
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.9
    max_tokens: int = 8192
    token_limit: int = 32768
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 42
    top_k: int = 20
    chat_template_kwargs: JsonDict = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    startup_timeout: int = 1800
    parallel_startup: bool = False
    startup_stagger_seconds: float = 0.0
    api_concurrency: int = 1
    api_timeout: float = 120.0
    api_max_retries: int = 3
    api_retry_sleep: float = 2.0
    api_input_token_limit: int = -1
    api_tokenizer_path: str = ""

    def fingerprint_payload(self) -> JsonDict:
        return {
            "backend": self.backend,
            "name": self.name,
            "model_path": self.model_path,
            "api_base_url": self.api_base_url,
            "instances": self.instances,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_tokens": self.max_tokens,
            "token_limit": self.token_limit,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "top_k": self.top_k,
            "chat_template_kwargs": self.chat_template_kwargs,
        }


@dataclass(frozen=True)
class LeanSpec:
    mathlib_path: str
    backend: str = "subprocess"
    check_concurrency: int = 16
    worker_pool_size: int = 0
    temp_dir: Optional[Path] = None

    @property
    def pool_size(self) -> int:
        if self.worker_pool_size > 0:
            return int(self.worker_pool_size)
        return max(1, int(self.check_concurrency))


@dataclass(frozen=True)
class DatasetSpec:
    parquet_dir: Optional[Path] = None
    parquet_glob: str = "*.parquet"
    id_column: str = "id"
    question_column: str = "question"
    response_column: str = "response"
    stage1_jsonl: Optional[Path] = None
    stage2_jsonl: Optional[Path] = None
    limit: int = -1


@dataclass(frozen=True)
class ArtifactSpec:
    root: Path
    schema_version: str = "step-proof-v2"
    resume: bool = True
    force: bool = False


@dataclass(frozen=True)
class StageSpec:
    name: str
    prompt_name: str
    model: Optional[ModelSpec] = None
    context_mode: str = "parent_only"
    batch_size: int = 64
    retries: int = 3
    wait_ms: int = 200
    max_pending_validation_batches: int = 4
    extra: JsonDict = field(default_factory=dict)

    def fingerprint_payload(self) -> JsonDict:
        return {
            "name": self.name,
            "prompt_name": self.prompt_name,
            "context_mode": self.context_mode,
            "batch_size": self.batch_size,
            "retries": self.retries,
            "extra": self.extra,
            "model": self.model.fingerprint_payload() if self.model else None,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    artifact: ArtifactSpec
    dataset: DatasetSpec
    lean: Optional[LeanSpec] = None
    stages: List[StageSpec] = field(default_factory=list)
    command: List[str] = field(default_factory=list)
    git_commit: str = ""

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["artifact"]["root"] = str(self.artifact.root)
        if self.dataset.parquet_dir is not None:
            payload["dataset"]["parquet_dir"] = str(self.dataset.parquet_dir)
        if self.dataset.stage1_jsonl is not None:
            payload["dataset"]["stage1_jsonl"] = str(self.dataset.stage1_jsonl)
        if self.dataset.stage2_jsonl is not None:
            payload["dataset"]["stage2_jsonl"] = str(self.dataset.stage2_jsonl)
        if self.lean is not None and self.lean.temp_dir is not None:
            payload["lean"]["temp_dir"] = str(self.lean.temp_dir)
        return payload
