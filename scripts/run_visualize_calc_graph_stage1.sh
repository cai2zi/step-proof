#!/usr/bin/env bash
set -euo pipefail

# 可视化阶段一建图结果（graph-only），输出 html。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-python3}"
GRAPH_JSONL="${GRAPH_JSONL:-${STEP_PROOF_ROOT}/calc_runs/graphs.jsonl}"
OUT_HTML="${OUT_HTML:-${STEP_PROOF_ROOT}/calc_runs/graph_only_dag.html}"
INDEX="${INDEX:-0}"

exec "${PYTHON}" "${STEP_PROOF_ROOT}/visualize_calc_graph_stage1.py" \
  --graph-jsonl "${GRAPH_JSONL}" \
  --out-html "${OUT_HTML}" \
  --index "${INDEX}" \
  "$@"