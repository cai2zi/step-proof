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
CONFIG="${EXP_DIR}/configs/rollout/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON}"

cd "${STEP_PROOF_ROOT}"
"${PYTHON_BIN}" "${EXP_DIR}/src/run_rollout.py" --config "${CONFIG}" "$@"
"${PYTHON_BIN}" "${EXP_DIR}/src/flatten_rollouts_for_step_proof.py" --config "${CONFIG}" "$@"
