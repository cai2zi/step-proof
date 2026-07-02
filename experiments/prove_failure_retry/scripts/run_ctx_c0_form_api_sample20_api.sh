#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-ctx_c0_form_api_sample20_seed0_gpt5mini}"
SOURCE_EXP="${SOURCE_EXP:-${CZX_ROOT}/czx_work/step-proof/rollout_rerank_math_verify/outputs/step_proofs/step_proof_ctx_c0_form_api}"
SAMPLE_PROBLEMS="${SAMPLE_PROBLEMS:-20}"
SEED="${SEED:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-stratified_random}"
STRATIFY_BUCKET_STEP="${STRATIFY_BUCKET_STEP:-20}"
SAMPLES_PER_BUCKET="${SAMPLES_PER_BUCKET:-0}"

bash "${SCRIPT_DIR}/run_retry.sh" \
  "exp_name=${EXP_NAME}" \
  "source.exp_dir=${SOURCE_EXP}" \
  "run.sample_problems=${SAMPLE_PROBLEMS}" \
  "run.seed=${SEED}" \
  "run.sample_mode=${SAMPLE_MODE}" \
  "run.stratify_bucket_step=${STRATIFY_BUCKET_STEP}" \
  "run.samples_per_bucket=${SAMPLES_PER_BUCKET}" \
  "classification.backend=api" \
  "retry.backend=api" \
  "$@"
