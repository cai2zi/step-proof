#!/usr/bin/env bash
set -euo pipefail

# Stage 3: prove from Stage 2 graph-form output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-/opt/anaconda3/envs/lean4-czx/bin/python}"

INFILE="${INFILE:-${STEP_PROOF_ROOT}/calc_runs/stage2_results.jsonl}"
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/calc_runs/stage3_results.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/calc_runs/stage3_failed.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STEP_PROOF_ROOT}/calc_runs/stage3_ckpt}"
LIMIT="${LIMIT:--1}"

MATHLIB_PATH="${MATHLIB_PATH:-/data/czx/mathlib4}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-64}"
LEAN_TEMP_DIR="${LEAN_TEMP_DIR:-${STEP_PROOF_ROOT}/calc_runs/lean_jobs}"

GPUS="${GPUS:-4,5,6,7}"
PROVER_GPUS="${PROVER_GPUS:-6,7}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ID_SCHEMA_MODE="${ID_SCHEMA_MODE:-calc}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-200}"

PROVER_MODEL_PATH="${PROVER_MODEL_PATH:-/data/czx/models/Goedel-Prover-V2-8B}"
PROVER_TP="${PROVER_TP:-2}"
PROVER_MAX_TOKENS="${PROVER_MAX_TOKENS:-8192}"
PROVER_TOKEN_LIMIT="${PROVER_TOKEN_LIMIT:-32768}"
PROVER_TEMPERATURE="${PROVER_TEMPERATURE:-0.2}"
PROVER_RETRIES="${PROVER_RETRIES:-3}"
PROVE_BATCH_SIZE="${PROVE_BATCH_SIZE:-64}"

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
  --prover-gpus "${PROVER_GPUS}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --id-schema-mode "${ID_SCHEMA_MODE}" \
  --batch-wait-ms "${BATCH_WAIT_MS}" \
  --prover-model-path "${PROVER_MODEL_PATH}" \
  --prover-tensor-parallel-size "${PROVER_TP}" \
  --prover-max-tokens "${PROVER_MAX_TOKENS}" \
  --prover-token-limit "${PROVER_TOKEN_LIMIT}" \
  --prover-temperature "${PROVER_TEMPERATURE}" \
  --prover-retries "${PROVER_RETRIES}" \
  --prove-batch-size "${PROVE_BATCH_SIZE}" \
  "$@"
