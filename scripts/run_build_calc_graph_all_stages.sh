#!/usr/bin/env bash
set -euo pipefail

# 一键串行运行 Stage1 -> Stage2 -> Stage3。
#
# 用法：
#   ./scripts/run_build_calc_graph_all_stages.sh
#
# 可选环境变量：
#   SKIP_STAGE1=1   # 跳过 stage1
#   SKIP_STAGE2=1   # 跳过 stage2
#   SKIP_STAGE3=1   # 跳过 stage3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

run_stage() {
  local stage_name="$1"
  local script_path="$2"
  echo "========== ${stage_name} 开始 =========="
  "${script_path}"
  echo "========== ${stage_name} 完成 =========="
}

if [[ "${SKIP_STAGE1:-0}" != "1" ]]; then
  run_stage "Stage1" "${SCRIPT_DIR}/run_build_calc_graph_stage1.sh"
else
  echo "跳过 Stage1 (SKIP_STAGE1=1)"
fi

if [[ "${SKIP_STAGE2:-0}" != "1" ]]; then
  run_stage "Stage2" "${SCRIPT_DIR}/run_build_calc_graph_stage2.sh"
else
  echo "跳过 Stage2 (SKIP_STAGE2=1)"
fi

if [[ "${SKIP_STAGE3:-0}" != "1" ]]; then
  run_stage "Stage3" "${SCRIPT_DIR}/run_build_calc_graph_stage3.sh"
else
  echo "跳过 Stage3 (SKIP_STAGE3=1)"
fi

echo "全部阶段执行完成。"
