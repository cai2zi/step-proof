#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B: graph model service
# 用法:
#   bash start_vllm_graph_qwen3.sh
# 可覆盖参数:
#   PORT=8001 TP=4 GPUS=0,1,2,3 MAX_LEN=8192 bash start_vllm_graph_qwen3.sh

MODEL_PATH="${MODEL_PATH:-/data/czx/models/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-40960}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

echo "[graph] model=${MODEL_PATH}"
echo "[graph] port=${PORT}, tp=${TP}, gpus=${GPUS}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --port "${PORT}"
