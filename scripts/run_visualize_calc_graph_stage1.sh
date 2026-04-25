#!/usr/bin/env bash
set -euo pipefail

# 可视化阶段一建图结果（graph-only），输出到目录，文件名为 <record_id>.html。
# 选择模式（全部 / 随机N / 指定ID列表）采用脚本内硬编码配置，不通过命令行传入。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="python3"
GRAPH_JSONL="${STEP_PROOF_ROOT}/result_stage1/graphs.jsonl"
OUT_DIR="${STEP_PROOF_ROOT}/result_stage1/HTML"
SEED="42"

# MODE: all | random | ids
MODE="all"

# MODE=random 时使用
RANDOM_N="3"

# MODE=ids 时使用（逗号分隔）
RECORD_IDS="16,6,11"

EXTRA_ARGS=()
if [[ "${MODE}" == "all" ]]; then
  EXTRA_ARGS+=(--all)
elif [[ "${MODE}" == "ids" ]]; then
  EXTRA_ARGS+=(--record-ids "${RECORD_IDS}")
elif [[ "${MODE}" == "random" ]]; then
  EXTRA_ARGS+=(--random-n "${RANDOM_N}")
else
  echo "Invalid MODE: ${MODE}. Expected 'all', 'random' or 'ids'." >&2
  exit 1
fi

exec "${PYTHON}" "${STEP_PROOF_ROOT}/visualize_calc_graph_stage1.py" \
  --graph-jsonl "${GRAPH_JSONL}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"