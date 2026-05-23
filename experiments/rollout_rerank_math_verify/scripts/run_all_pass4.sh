#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_pipeline_lib.sh
source "${SCRIPT_DIR}/run_pipeline_lib.sh"

CONFIG_NAME="base"
RUN_ROLLOUT=false

PIPELINE_OVERRIDES=(
  "rollout_config=base"
  "step_proof_config=base"
  "eval_config=base"
)

ROLLOUT_OVERRIDES=(
  "name=cases_reduce_prompt_API_pass4_correct_selected_wrong_all-rollouts"
)

STEP_PROOF_OVERRIDES=(
  "rollout_name=cases_reduce_prompt_API_pass4_correct_selected_wrong_all-rollouts"
  "name=reduce_prompt_API_pass4"
  "stage1.backend=api"
)

EVAL_OVERRIDES=(
  "rollout_name=cases_reduce_prompt_API_pass4_correct_selected_wrong_all-rollouts"
  "step_proof_name=reduce_prompt_API_pass4"
)

run_pipeline "${CONFIG_NAME}" "$@"
