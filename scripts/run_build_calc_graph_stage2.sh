#!/usr/bin/env bash
set -euo pipefail

# Stage 2: graph + form (local vLLM + concurrent Lean checks).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-/opt/anaconda3/envs/lean4-czx/bin/python}"

INFILE="${INFILE:-${STEP_PROOF_ROOT}/result_stage1/graphs.jsonl}"
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/result_stage2/stage2_results.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/result_stage2/stage2_failed.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STEP_PROOF_ROOT}/result_stage2/stage2_ckpt}"
LIMIT="${LIMIT:--1}"

MATHLIB_PATH="${MATHLIB_PATH:-/data/czx/mathlib4}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-64}"
LEAN_TEMP_DIR="${LEAN_TEMP_DIR:-${STEP_PROOF_ROOT}/result_stage2/lean_jobs}"

GPUS="${GPUS:-4,5,6,7}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ID_SCHEMA_MODE="${ID_SCHEMA_MODE:-calc}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-200}"

FORMALIZER_MODEL_PATH="${FORMALIZER_MODEL_PATH:-/data/czx/models/Goedel-Formalizer-V2-8B}"
FORMALIZER_TP="${FORMALIZER_TP:-4}"
FORMALIZER_MAX_TOKENS="${FORMALIZER_MAX_TOKENS:-8192}"
FORMALIZER_TOKEN_LIMIT="${FORMALIZER_TOKEN_LIMIT:-40960}"
FORMALIZER_TEMPERATURE="${FORMALIZER_TEMPERATURE:-0.2}"
FORMALIZER_RETRIES="${FORMALIZER_RETRIES:-3}"
FORM_BATCH_SIZE="${FORM_BATCH_SIZE:-64}"

exec "${PYTHON}" "${STEP_PROOF_ROOT}/build_calc_graph_stage2.py" \
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
  --formalizer-model-path "${FORMALIZER_MODEL_PATH}" \
  --formalizer-tensor-parallel-size "${FORMALIZER_TP}" \
  --formalizer-max-tokens "${FORMALIZER_MAX_TOKENS}" \
  --formalizer-token-limit "${FORMALIZER_TOKEN_LIMIT}" \
  --formalizer-temperature "${FORMALIZER_TEMPERATURE}" \
  --formalizer-retries "${FORMALIZER_RETRIES}" \
  --form-batch-size "${FORM_BATCH_SIZE}" \
  "$@"
