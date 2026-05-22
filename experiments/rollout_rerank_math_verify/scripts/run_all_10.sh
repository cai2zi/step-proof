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
  "rollout_name=qwen3_8b_10_temp07"
  "step_proof_name=reduce_prompt_API_10_temp07"
  "stage1_backend=api"
)

ROLLOUT_OVERRIDES=(
  "rollout.n=10"
  "rollout.temperature=0.7"
)

STEP_PROOF_OVERRIDES=(
  "stage2.gpus=1"
  "stage3.gpus=2"
  "stage2.formalizer_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Formalizer-V2-32B"
  "stage2.formalizer_retries=1"
)

EVAL_OVERRIDES=(
  "math_verify.random_seeds=[0,1,2,3]"
)

run_pipeline "${CONFIG_NAME}" "$@"
