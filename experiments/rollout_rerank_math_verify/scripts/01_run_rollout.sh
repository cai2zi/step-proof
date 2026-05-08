#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG="${1:-${EXP_DIR}/configs/experiment.yaml}"
PYTHON="${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}"

cd "${STEP_PROOF_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/run_rollout.py" --config "${CONFIG}"
