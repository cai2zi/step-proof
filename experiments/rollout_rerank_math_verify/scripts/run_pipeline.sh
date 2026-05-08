#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG="${1:-${EXP_DIR}/configs/experiment.yaml}"
LEAN4_PYTHON="${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}"

cd "${STEP_PROOF_ROOT}"
"${LEAN4_PYTHON}" "${SCRIPT_DIR}/00_prepare_input.py" --config "${CONFIG}"
bash "${SCRIPT_DIR}/01_run_rollout.sh" "${CONFIG}"
"${LEAN4_PYTHON}" "${SCRIPT_DIR}/02_flatten_rollouts_for_step_proof.py" --config "${CONFIG}"
bash "${SCRIPT_DIR}/03_run_step_proof.sh" "${CONFIG}"
"${LEAN4_PYTHON}" "${SCRIPT_DIR}/04_score_step_proof.py" --config "${CONFIG}"
"${LEAN4_PYTHON}" "${SCRIPT_DIR}/05_build_math_verify_inputs.py" --config "${CONFIG}"
bash "${SCRIPT_DIR}/06_run_math_verify.sh" "${CONFIG}"
"${LEAN4_PYTHON}" "${SCRIPT_DIR}/07_summarize_results.py" --config "${CONFIG}"
