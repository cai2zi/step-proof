#!/usr/bin/env bash
set -euo pipefail

if [[ "${RL_STOP_RAY_BEFORE_RUN:-1}" != "0" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

export CZX_ROOT="${CZX_ROOT:-/data/run01/scyb202/czx}"
export LEAN4_PYTHON="${LEAN4_PYTHON:-/data/home/scyb202/.conda/envs/lean4-czx/bin/python}"
PYTHON="${PYTHON:-${LEAN4_PYTHON}}"
export STEP_PROOF_RL_SCHED_TRACE=1 
CONFIG_PATH="${1:-configs/rl/fdg_grpo.yaml}"
shift || true

"${PYTHON}" scripts/rl/run_fdg_grpo.py --config "$CONFIG_PATH" "$@"
