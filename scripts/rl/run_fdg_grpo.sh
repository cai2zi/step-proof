#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/rl/fdg_grpo.yaml}"
shift || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python scripts/rl/run_fdg_grpo.py --config "$CONFIG_PATH" "$@"
