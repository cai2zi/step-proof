#!/usr/bin/env bash
set -euo pipefail

# 可视化阶段二结果，输出到 calc_runs/HTML，文件名为 <record_id>.html。
# 选择模式（随机N / 指定ID列表）采用脚本内硬编码配置，不通过命令行传入。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

# ── 硬编码配置区（按需修改） ───────────────────────────────────────────────
PYTHON="python3"
STAGE2_JSONL="${STEP_PROOF_ROOT}/result_stage3/stage3_results.jsonl"
OUT_DIR="/data/czx/step-proof/result_stage3/HTML"
# STAGE2_JSONL="${STEP_PROOF_ROOT}/result_stage2/stage2_results.jsonl"
# OUT_DIR="/data/czx/step-proof/result_stage2/HTML"
SOURCE="results"   # results | graph
GRAPH_ONLY="0"     # 1 => 使用 graph-only 视图
SEED="42"

# MODE: random | ids
MODE="ids"

# MODE=random 时使用
RANDOM_N="3"

# MODE=ids 时使用（逗号分隔）
RECORD_IDS="16,6,11,82,91,107,118,124,63"
# ──────────────────────────────────────────────────────────────────────────

EXTRA_ARGS=()
if [[ "${MODE}" == "ids" ]]; then
  EXTRA_ARGS+=(--record-ids "${RECORD_IDS}")
elif [[ "${MODE}" == "random" ]]; then
  EXTRA_ARGS+=(--random-n "${RANDOM_N}")
else
  echo "Invalid MODE: ${MODE}. Expected 'random' or 'ids'." >&2
  exit 1
fi

if [[ "${GRAPH_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--graph-only)
fi

exec "${PYTHON}" "${STEP_PROOF_ROOT}/visualize_calc_graph_stage2.py" \
  --stage2-jsonl "${STAGE2_JSONL}" \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
