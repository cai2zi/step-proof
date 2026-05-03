#!/usr/bin/env bash
set -euo pipefail

if [[ "${RL_STOP_RAY_BEFORE_RUN:-1}" != "0" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

export STEP_PROOF_RL_SCHED_TRACE=1 
CONFIG_PATH="${1:-configs/rl/fdg_grpo.yaml}"
shift || true

python scripts/rl/run_fdg_grpo.py --config "$CONFIG_PATH" "$@"
