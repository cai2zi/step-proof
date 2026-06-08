#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  run_ctx_c1_form_qwen3_8b.sh
  run_ctx_c1_form_qwen3_32b.sh
  # run_ctx_c1_form_api.sh
  run_ctx_c2_form_qwen3_8b.sh
  run_ctx_c2_form_qwen3_32b.sh
  # run_ctx_c2_form_api.sh
  run_ctx_c3_form_qwen3_8b.sh
  run_ctx_c3_form_qwen3_32b.sh
  # run_ctx_c3_form_api.sh
  run_ctx_c4_form_qwen3_8b.sh
  run_ctx_c4_form_qwen3_32b.sh
  # run_ctx_c4_form_api.sh
)

for script in "${SCRIPTS[@]}"; do
  echo "[ablation] running ${script}"
  bash "${SCRIPT_DIR}/${script}" "$@"
done
