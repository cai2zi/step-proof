#!/usr/bin/env bash
set -euo pipefail

# 阶段一建图（本地 vLLM，batch sliding-pool）。
# 已内置默认参数，可直接执行；也可追加命令行参数覆盖，或通过环境变量覆盖。
#
# 示例:
#   ./scripts/run_build_calc_graph_stage1.sh
#   ./scripts/run_build_calc_graph_stage1.sh --limit 10
#   LIMIT=10 ./scripts/run_build_calc_graph_stage1.sh
#
# 指定 Python（可选）:
#   PYTHON=/opt/anaconda3/envs/lean4-czx/bin/python ./scripts/run_build_calc_graph_stage1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/lean4-czx/bin/python}"

# ── 输入 ──────────────────────────────────────────────────────────────────
# PARQUET_DIR="${PARQUET_DIR:-/data/czx/data_raw/ODA-Math-460k/data_1}"
PARQUET_DIR="${PARQUET_DIR:-/workspace/mnt/lxb_work/czx_work/data_raw/ODA-extra}"
PARQUET_GLOB="${PARQUET_GLOB:-*.parquet}"
ID_COLUMN="${ID_COLUMN:-id}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
RESPONSE_COLUMN="${RESPONSE_COLUMN:-response}"
LIMIT="${LIMIT:-1000}"

# ── 输出 ──────────────────────────────────────────────────────────────────
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/result_stage1/graphs.jsonl}"
SKIPPED_JSONL="${SKIPPED_JSONL:-${STEP_PROOF_ROOT}/result_stage1/skipped.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/result_stage1/failed.jsonl}"

# ── vLLM ─────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-32B}"
TP="${TP:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-20}"
TOKEN_LIMIT="${TOKEN_LIMIT:-40960}"
# 可选：覆盖 chat 模板参数，例如 CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":true}'
# CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\":true}}"
CHAT_KWARGS_ARGS=()
if [ -n "${CHAT_TEMPLATE_KWARGS_JSON}" ]; then
  CHAT_KWARGS_ARGS=(--chat-template-kwargs-json "${CHAT_TEMPLATE_KWARGS_JSON}")
fi

# ── Batch / retry ─────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_RETRIES="${MAX_RETRIES:-3}"
GRAPH_MODE="${GRAPH_MODE:-legacy}"
INCLUDE_THINK_IN_DAG="${INCLUDE_THINK_IN_DAG:-0}"

# 仅支持 0/1：1 开启，其它值一律视为 0。
THINK_FLAG="--no-include-think-in-dag"
if [[ "${INCLUDE_THINK_IN_DAG}" == "1" ]]; then
  THINK_FLAG="--include-think-in-dag"
fi

exec "${PYTHON}" "${STEP_PROOF_ROOT}/build_calc_graph_stage1.py" \
  --parquet-dir    "${PARQUET_DIR}" \
  --glob           "${PARQUET_GLOB}" \
  --id-column      "${ID_COLUMN}" \
  --question-column "${QUESTION_COLUMN}" \
  --response-column "${RESPONSE_COLUMN}" \
  --limit          "${LIMIT}" \
  --out            "${OUT_JSONL}" \
  --skipped        "${SKIPPED_JSONL}" \
  --failed         "${FAILED_JSONL}" \
  --model-path     "${MODEL_PATH}" \
  --tensor-parallel-size "${TP}" \
  --gpus           "${GPUS}" \
  --dtype          "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-tokens     "${MAX_TOKENS}" \
  --temperature    "${TEMPERATURE}" \
  --top-p          "${TOP_P}" \
  --presence-penalty "${PRESENCE_PENALTY}" \
  --frequency-penalty "${FREQUENCY_PENALTY}" \
  --seed           "${SEED}" \
  --top-k          "${TOP_K}" \
  --token-limit    "${TOKEN_LIMIT}" \
  "${CHAT_KWARGS_ARGS[@]}" \
  --batch-size     "${BATCH_SIZE}" \
  --max-retries    "${MAX_RETRIES}" \
  --graph-mode     "${GRAPH_MODE}" \
  "${THINK_FLAG}" \
  "$@"
