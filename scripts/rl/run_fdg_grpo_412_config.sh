#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/rl/fdg_grpo_412.yaml}"
shift || true
ray stop
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_ROOT="${PROJECT_ROOT}/results"
EXPERIMENT_NAME="fdg_builder_grpo"
EXPERIMENT_ROOT="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
BUILDER_GPUS="4,5,6,7"
NUM_BUILDER_GPUS=4

export STEP_PROOF_RL_TRAIN_TRACE="${STEP_PROOF_RL_TRAIN_TRACE:-1}"
export STEP_PROOF_RL_SCHED_TRACE="${STEP_PROOF_RL_SCHED_TRACE:-1}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RL_SET_RAY_NUM_GPUS=0
export RL_STOP_RAY_BEFORE_RUN=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ "${RL_TEE_TERMINAL_LOG:-1}" != "0" ]]; then
  LOG_DIR="${EXPERIMENT_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/terminal_$(date +%Y-%m-%d_%H-%M-%S).log"
  echo "[run_fdg_grpo_412_config] writing terminal log to ${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

if [[ "${RL_STOP_RAY_BEFORE_START:-1}" != "0" ]]; then
  ray stop --force >/dev/null 2>&1 || true
  pkill -f "ray::" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  pkill -f "llm-worker-rl_" >/dev/null 2>&1 || true
fi

CUDA_VISIBLE_DEVICES="${BUILDER_GPUS}" \
  ray start --head \
  --num-gpus="${NUM_BUILDER_GPUS}" \
  --disable-usage-stats \
  --include-dashboard=false >/dev/null

cd "${PROJECT_ROOT}"
python scripts/rl/run_fdg_grpo.py --config "${CONFIG_PATH}" "$@"
