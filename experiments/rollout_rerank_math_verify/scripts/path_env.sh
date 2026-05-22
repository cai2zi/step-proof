#!/usr/bin/env bash

export CZX_ROOT="${CZX_ROOT:-/data/run01/scyb202/czx}"
export LEAN4_PYTHON="${LEAN4_PYTHON:-/data/home/scyb202/.conda/envs/lean4-czx/bin/python}"
export PYTHON="${PYTHON:-${LEAN4_PYTHON}}"
export EVAL_PYTHON="${EVAL_PYTHON:-${LEAN4_PYTHON}}"

export PROJECT_ROOT="${PROJECT_ROOT:-${CZX_ROOT}/step-proof}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${CZX_ROOT}/czx_work/step-proof/rollout_rerank_math_verify/outputs}"
export ROLLOUT_ROOT="${ROLLOUT_ROOT:-${OUTPUT_ROOT}/rollouts}"
export STEP_PROOF_OUTPUT_ROOT="${STEP_PROOF_OUTPUT_ROOT:-${OUTPUT_ROOT}/step_proofs}"
export MODEL_ROOT="${MODEL_ROOT:-${CZX_ROOT}/models}"
export DATA_ROOT="${DATA_ROOT:-${CZX_ROOT}/data_raw}"
export MATHLIB_ROOT="${MATHLIB_ROOT:-${CZX_ROOT}/mathlib4}"
export LEAN_TEMP_ROOT="${LEAN_TEMP_ROOT:-${CZX_ROOT}/czx_work/TEMP/lean_jobs}"

