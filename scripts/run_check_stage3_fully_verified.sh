#!/usr/bin/env bash
set -euo pipefail

# 统计 Stage3 中“所有节点均 Fully verified”的样本数。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-python3}"
STAGE3_JSONL="${STAGE3_JSONL:-${STEP_PROOF_ROOT}/result_stage3/stage3_results.jsonl}"
SHOW_IDS="${SHOW_IDS:-0}"

EXTRA_ARGS=()
if [[ "${SHOW_IDS}" == "1" ]]; then
  EXTRA_ARGS+=(--show-ids)
fi

exec "${PYTHON}" "${STEP_PROOF_ROOT}/check_stage3_fully_verified.py" \
  --stage3-jsonl "${STAGE3_JSONL}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
