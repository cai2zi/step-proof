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
  "rollout_name=cases_reduce_prompt_API_pass4_correct_selected_wrong_all-rollouts"
  "step_proof_name=reduce_prompt_API_pass4_11"
  "stage1_backend=api"
)

ROLLOUT_OVERRIDES=()

STEP_PROOF_OVERRIDES=(
  "stage2.formalizer_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Formalizer-V2-32B"
  "stage3.prover_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Prover-V2-32B"
)

EVAL_OVERRIDES=()

run_pipeline "${CONFIG_NAME}" "$@"
