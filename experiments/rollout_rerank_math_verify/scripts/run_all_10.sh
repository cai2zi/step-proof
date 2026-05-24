#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_pipeline_lib.sh
source "${SCRIPT_DIR}/run_pipeline_lib.sh"

CONFIG_NAME="base"
RUN_ROLLOUT=true

PIPELINE_OVERRIDES=(
  "rollout_config=base"
  "step_proof_config=base"
  "eval_config=base"
  "gpus=0,1"
)

ROLLOUT_OVERRIDES=(
  "name=qwen3_8b_except_gsm8k"
)

STEP_PROOF_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "name=reduce_prompt_API_10"
  
  
  "stage1.backend=api"
  "stage1.reuse_from_step_proof=reduce_prompt_API"
  "stage2.formalizer_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Formalizer-V2-32B"
)

EVAL_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "step_proof_name=reduce_prompt_API_10"
)

run_pipeline "${CONFIG_NAME}" "$@"
