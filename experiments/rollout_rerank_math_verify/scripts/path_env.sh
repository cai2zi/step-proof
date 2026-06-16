#!/usr/bin/env bash
#d:\program\env\python\envs\graph
export CZX_ROOT="${CZX_ROOT:-/data/czx}"
export LEAN4_PYTHON="${LEAN4_PYTHON:-/opt/anaconda3/envs/lean4-czx/bin/python}"
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

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CZX_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TMPDIR="${TMPDIR:-${CZX_ROOT}/TEMP}"


export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1

mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${VLLM_CACHE_ROOT}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${TMPDIR}"

export ELAN_HOME=/data/run01/scyb202/czx/.elan
export CARGO_HOME=/data/run01/scyb202/czx/.cargo
export PATH="$ELAN_HOME/bin:$CARGO_HOME/bin:$PATH"