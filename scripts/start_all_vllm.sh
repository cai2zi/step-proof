#!/usr/bin/env bash
set -euo pipefail

# 一键启动三个 vLLM 服务（后台运行）
# 输出:
#   logs/vllm_*.log
#   logs/vllm_*.pid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

start_service() {
  local name="$1"
  local script="$2"
  local log_file="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  echo "[start] ${name} -> ${log_file}"
  nohup bash "${SCRIPT_DIR}/${script}" > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  echo "[ok] ${name} pid=${pid}"
}

start_service "vllm_graph" "start_vllm_graph_qwen3.sh"
start_service "vllm_formalizer" "start_vllm_formalizer.sh"
start_service "vllm_prover" "start_vllm_prover.sh"

echo
echo "全部已提交后台。检查方式:"
echo "  ps -fp \$(cat \"${LOG_DIR}\"/*.pid)"
echo "  tail -f \"${LOG_DIR}\"/vllm_graph.log"
echo "  tail -f \"${LOG_DIR}\"/vllm_formalizer.log"
echo "  tail -f \"${LOG_DIR}\"/vllm_prover.log"
