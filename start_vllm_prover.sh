#!/usr/bin/env bash
set -euo pipefail

# Goedel-Prover-V2-8B service
# 用法:
#   bash start_vllm_prover.sh
# 可覆盖参数:
#   PORT=8003 TP=2 GPUS=4,5 MAX_LEN=8192 bash start_vllm_prover.sh

MODEL_PATH="${MODEL_PATH:-/data/czx/models/Goedel-Prover-V2-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-goedel-prover-v2-8b}"
PORT="${PORT:-8003}"
TP="${TP:-2}"
GPUS="${GPUS:-4,5}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

echo "[prover] model=${MODEL_PATH}"
echo "[prover] port=${PORT}, tp=${TP}, gpus=${GPUS}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --port "${PORT}"
