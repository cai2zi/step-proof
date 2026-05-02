#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/rl/fdg_grpo.yaml}"
shift || true

python scripts/rl/run_fdg_grpo.py --config "$CONFIG_PATH" "$@"
