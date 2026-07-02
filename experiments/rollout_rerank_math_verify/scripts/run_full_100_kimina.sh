#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_pipeline_lib.sh
source "${SCRIPT_DIR}/run_pipeline_lib.sh"

CONFIG_NAME="base"
RUN_ROLLOUT=false

STEP_PROOF_NAME="${STEP_PROOF_NAME:-full_100_kimina}"
KIMINA_API_URL="${KIMINA_API_URL:-http://localhost:8000}"
KIMINA_API_KEY_ENV="${KIMINA_API_KEY_ENV:-KIMINA_API_KEY}"
KIMINA_LEAN_TIMEOUT="${KIMINA_LEAN_TIMEOUT:-300}"

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
  "name=${STEP_PROOF_NAME}"

  # runtime config
  "stage1.backend=api"
  "stage2.formalizer_instances=2"
  "stage3.prover_instances=2"
  "lean_runtime.lean_backend=kimina_server"
  "$(hydra_string_override lean_runtime.lean_api_url "${KIMINA_API_URL}")"
  "lean_runtime.lean_api_key_env=${KIMINA_API_KEY_ENV}"
  "lean_runtime.lean_server_timeout=${KIMINA_LEAN_TIMEOUT}"

  # experiment config
  "stage1.fdg_prompt=fdg_full_graph"
  "stage1.validation_checks.all_facts_reach_answer=false"
  "stage2.formalizer_model_path=\${oc.env:CZX_ROOT}/models/Goedel-Formalizer-V2-8B"
)

EVAL_OVERRIDES=(
  "rollout_name=qwen3_8b_except_gsm8k"
  "step_proof_name=${STEP_PROOF_NAME}"
)

run_pipeline "${CONFIG_NAME}" "$@"
