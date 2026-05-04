from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RewardWeights:
    invalid_json_penalty: float = -1.0
    invalid_fdg_penalty: float = -0.6
    valid_json_bonus: float = 0.05
    validator_pass_bonus: float = 0.15
    formalizer_pass_weight: float = 0.4
    prover_pass_weight: float = 0.2
    final_answer_pass_weight: float = 0.2
    warning_penalty: float = 0.01


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_path: str
    gpus: str
    tensor_parallel_size: int
    max_tokens: int
    token_limit: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    seed: int
    top_k: int
    retries: int
    batch_size: int
    prompt_name: str
    num_workers: int = 1
    gpu_memory_utilization: float = 0.9
    chat_template_kwargs: JsonDict = field(default_factory=lambda: {"enable_thinking": False})


@dataclass(frozen=True)
class LeanRuntimeConfig:
    mathlib_path: str
    backend: str
    check_concurrency: int
    worker_pool_size: int
    temp_dir: str


@dataclass(frozen=True)
class SchedulerRuntimeConfig:
    graph_wait_ms: int = 50
    max_graph_batch_size: int = 64
    formalizer_wait_ms: int = 25
    prover_wait_ms: int = 100
    max_pending_graphs: int = 4096
    runtime_actor_max_concurrency: int = 128


@dataclass(frozen=True)
class TraceRuntimeConfig:
    enabled: bool = False
    out_dir: str = "results/fdg_builder_grpo/cot_traces"


@dataclass(frozen=True)
class FDGRLEvaluatorConfig:
    weights: RewardWeights
    formalizer: ModelRuntimeConfig
    prover: ModelRuntimeConfig
    lean: LeanRuntimeConfig
    scheduler: SchedulerRuntimeConfig = field(default_factory=SchedulerRuntimeConfig)
    trace: TraceRuntimeConfig = field(default_factory=TraceRuntimeConfig)
    include_prover: bool = True


@dataclass(frozen=True)
class CandidateGraphInput:
    record_id: str
    problem_text: str
    solution_or_cot: str
    model_output: str
    ground_truth: str = ""
    data_source: str = "fdg_builder"
    extra_info: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeFactTask:
    sample_index: int
    fact: JsonDict
    attempt: int = 1
    feedback_messages: List[JsonDict] = field(default_factory=list)
    attempt_history: List[JsonDict] = field(default_factory=list)


@dataclass
class BridgeGenerationResult:
    sample_index: int
    fact_id: str
    stage: str
    attempts: int
    extracted: bool
    lean_code: str
    error_message: str
    raw_output: str = ""
    attempt_history: List[JsonDict] = field(default_factory=list)
    conversation: Optional[List[JsonDict]] = None


@dataclass
class BridgeFactResult:
    sample_index: int
    fact_id: str
    stage: str
    attempts: int
    success: bool
    verified: bool
    lean_code: str
    error_message: str
    raw_output: str = ""
    attempt_history: List[JsonDict] = field(default_factory=list)
    conversation: Optional[List[JsonDict]] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class FactRewardTrace:
    fact_id: str
    text: str
    parent_fact_ids: List[str]
    is_final_answer: bool
    origin: str
    proof_obligation: JsonDict = field(default_factory=dict)
    formalizer: Optional[BridgeFactResult] = None
    prover: Optional[BridgeFactResult] = None

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        if self.formalizer is not None:
            payload["formalizer"] = self.formalizer.to_dict()
        if self.prover is not None:
            payload["prover"] = self.prover.to_dict()
        return payload


@dataclass
class GraphRewardBreakdown:
    record_id: str
    score: float
    structure_score: float
    formalizer_score: float
    prover_score: float
    final_answer_score: float
    length_penalty: float
    valid_json: bool
    validator_passed: bool
    num_facts: int
    num_non_root_facts: int
    num_final_facts: int
    num_warnings: int
    num_formalized: int
    num_proved: int
    num_final_verified: int
    errors: List[JsonDict] = field(default_factory=list)
    warnings: List[JsonDict] = field(default_factory=list)
    facts: List[FactRewardTrace] = field(default_factory=list)
    parse_error: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "record_id": self.record_id,
            "score": self.score,
            "structure_score": self.structure_score,
            "formalizer_score": self.formalizer_score,
            "prover_score": self.prover_score,
            "final_answer_score": self.final_answer_score,
            "length_penalty": self.length_penalty,
            "valid_json": self.valid_json,
            "validator_passed": self.validator_passed,
            "num_facts": self.num_facts,
            "num_non_root_facts": self.num_non_root_facts,
            "num_final_facts": self.num_final_facts,
            "num_warnings": self.num_warnings,
            "num_formalized": self.num_formalized,
            "num_proved": self.num_proved,
            "num_final_verified": self.num_final_verified,
            "errors": self.errors,
            "warnings": self.warnings,
            "facts": [fact.to_dict() for fact in self.facts],
            "parse_error": self.parse_error,
        }

    def to_reward_dict(self, *, include_trace: bool = False) -> JsonDict:
        payload = {
            "score": self.score,
            "validator_passed": self.validator_passed,
            "valid_json": self.valid_json,
            "structure_score": self.structure_score,
            "formalizer_score": self.formalizer_score,
            "prover_score": self.prover_score,
            "final_answer_score": self.final_answer_score,
            "length_penalty": self.length_penalty,
            "num_facts": self.num_facts,
            "num_non_root_facts": self.num_non_root_facts,
            "num_formalized": self.num_formalized,
            "num_proved": self.num_proved,
            "num_final_verified": self.num_final_verified,
            "num_warnings": self.num_warnings,
            "parse_error": self.parse_error,
        }
        if include_trace:
            payload["fdg_reward_trace"] = self.to_dict()
        return payload
