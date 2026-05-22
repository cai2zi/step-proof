#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=path_env.sh
source "${SCRIPT_DIR}/path_env.sh"

"${PYTHON}" "${SCRIPT_DIR}/prepare_single_debug_rollout.py" \
math_500_test__test_precalculus_625.json \
--all-rollouts 
