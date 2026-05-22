#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=path_env.sh
source "${SCRIPT_DIR}/path_env.sh"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-base}"
if [[ $# -gt 0 ]]; then
  shift
fi
PYTHON_BIN="${PYTHON}"

cd "${STEP_PROOF_ROOT}"
PYTHON="${PYTHON_BIN}" "${STEP_PROOF_ROOT}/scripts/run_experiment.sh" \
  --config-path "${EXP_DIR}/configs/step_proof" \
  --config-name "${CONFIG_NAME}" \
  "$@"
