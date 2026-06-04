#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/02_step_proof.sh" base \
  "rollout_name=qwen3_8b_except_gsm8k" \
  "name=formalizer_ctx_c1" \
  "run.stages=[stage1,stage2,stage3,stats]" \
  "stage1.reuse_from_step_proof=full_100" \
  "stage1.reuse_require_all=true" \
  "stage1.fdg_prompt=fdg_full_graph" \
  "stage1.validation_checks.all_facts_reach_answer=false" \
  "stage2.backend=api" \
  "stage2.api_input_token_limit=16384" \
  "stage2.formalizer_prompt=formalize_obligation.api_context" \
  "stage2.formalizer_context_mode=c1_problem_parent" \
  "$@"
