#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CZX_ROOT="${CZX_ROOT:-/data/run01/scyb202/czx}"
export LEAN4_PYTHON="${LEAN4_PYTHON:-/data/home/scyb202/.conda/envs/lean4-czx/bin/python}"
export PYTHON="${PYTHON:-${LEAN4_PYTHON}}"
export KIMINA_API_URL="${KIMINA_API_URL:-http://localhost:8000}"

CONFIG="${CONFIG:-${EXP_DIR}/configs/base.yaml}"

"${PYTHON}" "${EXP_DIR}/scripts/run_retry_experiment.py" \
  --config "${CONFIG}" \
  "$@"
