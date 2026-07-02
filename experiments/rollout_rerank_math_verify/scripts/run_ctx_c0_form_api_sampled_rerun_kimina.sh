#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_pipeline_lib.sh
source "${SCRIPT_DIR}/run_pipeline_lib.sh"

CONFIG_NAME="base"
RUN_ROLLOUT=false

SOURCE_ROLLOUT_NAME="${SOURCE_ROLLOUT_NAME:-qwen3_8b_except_gsm8k}"
SAMPLED_ROLLOUT_NAME="${SAMPLED_ROLLOUT_NAME:-qwen3_8b_except_gsm8k_sampled_ctx_c0_form_api_seed0}"
STEP_PROOF_NAME="${STEP_PROOF_NAME:-ctx_c0_form_api_sampled_rerun_kimina_seed0}"
RUN_EVAL="${RUN_EVAL:-false}"
KIMINA_API_URL="${KIMINA_API_URL:-http://localhost:8000}"
KIMINA_API_KEY_ENV="${KIMINA_API_KEY_ENV:-KIMINA_API_KEY}"
KIMINA_LEAN_TIMEOUT="${KIMINA_LEAN_TIMEOUT:-300}"

PIPELINE_OVERRIDES=(
  "rollout_config=base"
  "step_proof_config=base"
  "eval_config=base"
  "gpus=0,1,2,3"
)

ROLLOUT_OVERRIDES=(
  "name=${SOURCE_ROLLOUT_NAME}"
)

STEP_PROOF_OVERRIDES=(
  "rollout_name=${SAMPLED_ROLLOUT_NAME}"
  "name=${STEP_PROOF_NAME}"
  "run.stages=[stage1,stage2,stage3,stats]"
  "stage1.fdg_prompt=fdg_full_graph"
  "stage1.validation_checks.all_facts_reach_answer=false"
  "stage2.backend=api"
  "stage2.formalizer_prompt=formalize_obligation.api_context"
  "stage2.formalizer_context_mode=c0_parent"
  "lean_runtime.lean_backend=kimina_server"
  "$(hydra_string_override lean_runtime.lean_api_url "${KIMINA_API_URL}")"
  "lean_runtime.lean_api_key_env=${KIMINA_API_KEY_ENV}"
  "lean_runtime.lean_server_timeout=${KIMINA_LEAN_TIMEOUT}"
)

EVAL_OVERRIDES=(
  "rollout_name=${SAMPLED_ROLLOUT_NAME}"
  "step_proof_name=${STEP_PROOF_NAME}"
)

run_pipeline "${CONFIG_NAME}" "$@"
