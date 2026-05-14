#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-qwen3_8b}"
CONFIG="${EXP_DIR}/configs/rollout/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON:-${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}}"

cd "${STEP_PROOF_ROOT}"
"${PYTHON_BIN}" "${EXP_DIR}/src/run_rollout.py" --config "${CONFIG}"
"${PYTHON_BIN}" "${EXP_DIR}/src/flatten_rollouts_for_step_proof.py" --config "${CONFIG}"
