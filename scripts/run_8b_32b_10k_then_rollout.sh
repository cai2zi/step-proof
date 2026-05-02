#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${STEP_PROOF_ROOT}/.." && pwd)"

PYTHON="${PYTHON:-python}"
CONFIG_NAME="${CONFIG_NAME:-experiment_fdg}"

run_exp() {
  local exp_name="$1"
  local model_path="$2"
  shift 2

  echo "==> Running experiment: ${exp_name}"
  echo "    stage1.model_path=${model_path}"
  echo "    stage1.limit=10000"

  "${PYTHON}" "${STEP_PROOF_ROOT}/run_experiment.py" \
    --config-name "${CONFIG_NAME}" \
    "exp.name=\"${exp_name}\"" \
    "stage1.model_path=${model_path}" \
    "stage1.limit=10000" \
    "$@"
}

run_exp "qwen8B_10k" "/root/autodl-tmp/models/Qwen3-8B"
run_exp "qwen32B_10k" "/root/autodl-tmp/models/Qwen3-32B"

echo "==> Running rollout"
"${WORKSPACE_ROOT}/llm-infer/scripts/rollout.sh"
