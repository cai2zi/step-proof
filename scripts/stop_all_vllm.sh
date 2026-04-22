#!/usr/bin/env bash
set -euo pipefail

# 停止 start_all_vllm.sh 启动的三个 vLLM 服务
# 依赖文件:
#   logs/vllm_graph.pid
#   logs/vllm_formalizer.pid
#   logs/vllm_prover.pid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

stop_service() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[skip] ${name}: pid 文件不存在 (${pid_file})"
    return
  fi

  local pid
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "[skip] ${name}: pid 文件为空"
    rm -f "${pid_file}"
    return
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[skip] ${name}: 进程 ${pid} 不存在"
    rm -f "${pid_file}"
    return
  fi

  echo "[stop] ${name}: 发送 SIGTERM 到 pid=${pid}"
  kill "${pid}" 2>/dev/null || true

  # 最多等待 10 秒优雅退出
  for _ in {1..10}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ok] ${name}: 已停止"
      rm -f "${pid_file}"
      return
    fi
    sleep 1
  done

  echo "[force] ${name}: 超时，发送 SIGKILL 到 pid=${pid}"
  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
}

stop_service "vllm_graph"
stop_service "vllm_formalizer"
stop_service "vllm_prover"

echo "全部停止流程已完成。"
