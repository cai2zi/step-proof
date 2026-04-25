#!/usr/bin/env bash
set -euo pipefail

# Stage 3: prove from Stage 2 graph-form output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/lean4-czx/bin/python}"

INFILE="${INFILE:-${STEP_PROOF_ROOT}/result_stage2/stage2_results.jsonl}"
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/result_stage3/stage3_results.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/result_stage3/stage3_failed.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STEP_PROOF_ROOT}/result_stage3/stage3_ckpt}"
LIMIT="${LIMIT:--1}"

MATHLIB_PATH="${MATHLIB_PATH:-/workspace/mnt/lxb_work/czx_work/mathlib4}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-100}"
LEAN_TEMP_DIR="${LEAN_TEMP_DIR:-${STEP_PROOF_ROOT}/result_stage3/lean_jobs}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
ID_SCHEMA_MODE="${ID_SCHEMA_MODE:-calc}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-200}"
MAX_PENDING_VALIDATION_BATCHES="${MAX_PENDING_VALIDATION_BATCHES:-4}"
MAX_PENDING_VALIDATION_ITEMS="${MAX_PENDING_VALIDATION_ITEMS:-0}"

# PROVER_MODEL_PATH="${PROVER_MODEL_PATH:-/workspace/mnt/lxb_work/czx_work/models/Goedel-Prover-V2-32B}"
PROVER_MODEL_PATH="${PROVER_MODEL_PATH:-/workspace/mnt/lxb_work/czx_work/models/Goedel-Prover-V2-8B}"

PROVER_TP="${PROVER_TP:-8}"
PROVER_MAX_TOKENS="${PROVER_MAX_TOKENS:-8192}"
PROVER_TOKEN_LIMIT="${PROVER_TOKEN_LIMIT:-40960}"
PROVER_TEMPERATURE="${PROVER_TEMPERATURE:-0.0}"
PROVER_TOP_P="${PROVER_TOP_P:-1.0}"
PROVER_PRESENCE_PENALTY="${PROVER_PRESENCE_PENALTY:-0.0}"
PROVER_FREQUENCY_PENALTY="${PROVER_FREQUENCY_PENALTY:-0.0}"
PROVER_SEED="${PROVER_SEED:-42}"
PROVER_TOP_K="${PROVER_TOP_K:-20}"
PROVER_CHAT_TEMPLATE_KWARGS_JSON="${PROVER_CHAT_TEMPLATE_KWARGS_JSON:-}"
PROVER_CHAT_TEMPLATE_KWARGS_JSON="${PROVER_CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\":true}}"
PROVER_CHAT_KWARGS_ARGS=()
if [ -n "${PROVER_CHAT_TEMPLATE_KWARGS_JSON}" ]; then
  PROVER_CHAT_KWARGS_ARGS=(--prover-chat-template-kwargs-json "${PROVER_CHAT_TEMPLATE_KWARGS_JSON}")
fi

PROVER_RETRIES="${PROVER_RETRIES:-3}"
PROVE_BATCH_SIZE="${PROVE_BATCH_SIZE:-128}"

exec "${PYTHON}" "${STEP_PROOF_ROOT}/build_calc_graph_stage3.py" \
  --infile "${INFILE}" \
  --out "${OUT_JSONL}" \
  --failed "${FAILED_JSONL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --limit "${LIMIT}" \
  --mathlib-path "${MATHLIB_PATH}" \
  --lean-check-concurrency "${LEAN_CHECK_CONCURRENCY}" \
  --lean-temp-dir "${LEAN_TEMP_DIR}" \
  --gpus "${GPUS}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --id-schema-mode "${ID_SCHEMA_MODE}" \
  --batch-wait-ms "${BATCH_WAIT_MS}" \
  --max-pending-validation-batches "${MAX_PENDING_VALIDATION_BATCHES}" \
  --max-pending-validation-items "${MAX_PENDING_VALIDATION_ITEMS}" \
  --prover-model-path "${PROVER_MODEL_PATH}" \
  --prover-tensor-parallel-size "${PROVER_TP}" \
  --prover-max-tokens "${PROVER_MAX_TOKENS}" \
  --prover-token-limit "${PROVER_TOKEN_LIMIT}" \
  --prover-temperature "${PROVER_TEMPERATURE}" \
  --prover-top-p "${PROVER_TOP_P}" \
  --prover-presence-penalty "${PROVER_PRESENCE_PENALTY}" \
  --prover-frequency-penalty "${PROVER_FREQUENCY_PENALTY}" \
  --prover-seed "${PROVER_SEED}" \
  --prover-top-k "${PROVER_TOP_K}" \
  "${PROVER_CHAT_KWARGS_ARGS[@]}" \
  --prover-retries "${PROVER_RETRIES}" \
  --prove-batch-size "${PROVE_BATCH_SIZE}" \
  "$@"
