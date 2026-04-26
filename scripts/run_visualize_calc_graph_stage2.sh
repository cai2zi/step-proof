#!/usr/bin/env bash
set -euo pipefail

# 启动交互式前端页面：
# - 自动扫描 results/* 中含 stage3_results.jsonl 的实验名（单选）
# - 输入 record_id 后点击按钮，渲染并展示可视化 HTML

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

# ── 可配置项 ───────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
RESULTS_ROOT="${STEP_PROOF_ROOT}/results"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
SOURCE="${SOURCE:-results}"   # results | graph
GRAPH_ONLY="${GRAPH_ONLY:-0}" # 1 => 使用 graph-only 视图
# ──────────────────────────────────────────────────────────────────────────

EXTRA_ARGS=()
if [[ "${GRAPH_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--graph-only)
fi

echo "Starting interactive viewer at http://${HOST}:${PORT}"
exec "${PYTHON}" "${STEP_PROOF_ROOT}/scripts/interactive_stage3_viewer.py" \
  --results-root "${RESULTS_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --source "${SOURCE}" \
  "${EXTRA_ARGS[@]}"
