#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-fdg_bug}"
PYTHON_BIN="${PYTHON:-${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}}"

cd "${STEP_PROOF_ROOT}"
PYTHON="${PYTHON_BIN}" "${STEP_PROOF_ROOT}/scripts/run_experiment.sh" \
  --config-path "${EXP_DIR}/configs/step_proof" \
  --config-name "${CONFIG_NAME}"
