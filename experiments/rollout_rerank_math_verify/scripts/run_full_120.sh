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
  "gpus=0,1"
)

ROLLOUT_OVERRIDES=(
  "name=qwen3_8b_except_gsm8k"
  "rollout.instances=2"
)

STEP_PROOF_OVERRIDES=(
  # name
  "rollout_name=qwen3_8b_except_gsm8k"
  "name=full_120"

  # runtime config
  "stage1.backend=api"
  "stage2.backend=api"
  "stage3.prover_instances=2"

  # experiment config
  "stage1.reuse_from_step_proof=full_100"
  "stage1.reuse_require_all=true"
  "stage1.fdg_prompt=fdg_full_graph"
  "stage1.validation_checks.all_facts_reach_answer=false"
  "stage2.formalizer_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Formalizer-V2-8B"
)

EVAL_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "step_proof_name=full_120"
)

run_pipeline "${CONFIG_NAME}" "$@"
