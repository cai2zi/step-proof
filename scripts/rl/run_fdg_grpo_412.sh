#!/usr/bin/env bash
set -euo pipefail

export RL_STOP_RAY_BEFORE_RUN="${RL_STOP_RAY_BEFORE_RUN:-1}"
export RL_BUILDER_GPUS="${RL_BUILDER_GPUS:-3,4,5,6}"
export RL_NUM_GPUS="${RL_NUM_GPUS:-4}"
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-1}"
export RL_AGENT_NUM_WORKERS="${RL_AGENT_NUM_WORKERS:-4}"
export RL_FORMALIZER_GPUS="${RL_FORMALIZER_GPUS:-2}"
export RL_FORMALIZER_TP="${RL_FORMALIZER_TP:-1}"
export RL_PROVER_GPUS="${RL_PROVER_GPUS:-0,1}"
export RL_PROVER_TP="${RL_PROVER_TP:-1}"
export RL_PROVER_NUM_WORKERS="${RL_PROVER_NUM_WORKERS:-2}"
export RL_FORMALIZER_GPU_MEMORY_UTILIZATION="${RL_FORMALIZER_GPU_MEMORY_UTILIZATION:-0.95}"
export RL_PROVER_GPU_MEMORY_UTILIZATION="${RL_PROVER_GPU_MEMORY_UTILIZATION:-0.95}"
export RL_VAL_BEFORE_TRAIN="${RL_VAL_BEFORE_TRAIN:-0}"
export RL_TEST_FREQ="${RL_TEST_FREQ:--1}"
export RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-fdg_builder_grpo}"
export RL_RESULTS_ROOT="${RL_RESULTS_ROOT:-$(pwd)/results}"
export RL_EXPERIMENT_ROOT="${RL_EXPERIMENT_ROOT:-${RL_RESULTS_ROOT}/${RL_EXPERIMENT_NAME}}"
export RL_FDG_RUNTIME_ACTOR_NAME="${RL_FDG_RUNTIME_ACTOR_NAME:-fdg_rl_runtime_${RL_EXPERIMENT_NAME}}"
export RL_LEAN_TEMP_DIR="${RL_LEAN_TEMP_DIR:-${RL_EXPERIMENT_ROOT}/tmp/lean_jobs}"
export RL_FDG_COT_TRACE_DIR="${RL_FDG_COT_TRACE_DIR:-${RL_EXPERIMENT_ROOT}/cot_traces}"
export RL_FDG_COT_TRACE="${RL_FDG_COT_TRACE:-1}"
export RL_FDG_LOG_SAMPLES_PER_STEP="${RL_FDG_LOG_SAMPLES_PER_STEP:-16}"
export STEP_PROOF_RL_TRAIN_TRACE="${STEP_PROOF_RL_TRAIN_TRACE:-1}"

if [[ "${RL_TEE_TERMINAL_LOG:-1}" != "0" ]]; then
  LOG_DIR="${RL_EXPERIMENT_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/terminal_$(date +%Y-%m-%d_%H-%M-%S).log"
  echo "[run_fdg_grpo_412] writing terminal log to ${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

if [[ "${RL_STOP_RAY_BEFORE_RUN}" != "0" ]]; then
  ray stop --force >/dev/null 2>&1 || true
  pkill -f "ray::" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  pkill -f "llm-worker-rl_" >/dev/null 2>&1 || true
fi

CUDA_VISIBLE_DEVICES="${RL_BUILDER_GPUS}" \
  ray start --head \
  --num-gpus="${RL_NUM_GPUS}" \
  --disable-usage-stats \
  --include-dashboard=false >/dev/null

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RL_STOP_RAY_BEFORE_RUN=0
export RL_SET_RAY_NUM_GPUS=0

CONFIG_PATH="${1:-configs/rl/fdg_grpo.yaml}"
shift || true

bash scripts/rl/run_fdg_grpo.sh "$CONFIG_PATH" "$@"
