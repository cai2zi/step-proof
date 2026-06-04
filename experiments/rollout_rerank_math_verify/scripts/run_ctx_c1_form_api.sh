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
  "gpus=0,1,2,3"
)

ROLLOUT_OVERRIDES=(
  "name=qwen3_8b_except_gsm8k"
)

STEP_PROOF_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "name=ctx_c1_form_api"
  "run.stages=[stage1,stage2,stage3,stats]"
  "stage1.reuse_from_step_proof=full_100"
  "stage1.reuse_require_all=true"
  "stage1.fdg_prompt=fdg_full_graph"
  "stage1.validation_checks.all_facts_reach_answer=false"
  "stage2.backend=api"
  "stage2.formalizer_prompt=formalize_obligation.api_context"
  "stage2.formalizer_context_mode=c1_problem_parent"
)

EVAL_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "step_proof_name=ctx_c1_form_api"
)

run_pipeline "${CONFIG_NAME}" "$@"
