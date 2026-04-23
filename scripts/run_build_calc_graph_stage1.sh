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

PYTHON="${PYTHON:-/opt/anaconda3/envs/lean4-czx/bin/python}"

# ── 输入 ──────────────────────────────────────────────────────────────────
PARQUET_DIR="${PARQUET_DIR:-/data/czx/data_raw/ODA-Math-460k/data}"
PARQUET_GLOB="${PARQUET_GLOB:-*.parquet}"
ID_COLUMN="${ID_COLUMN:-id}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
RESPONSE_COLUMN="${RESPONSE_COLUMN:-response}"
LIMIT="${LIMIT:--1}"

# ── 输出 ──────────────────────────────────────────────────────────────────
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/result_stage1/graphs.jsonl}"
SKIPPED_JSONL="${SKIPPED_JSONL:-${STEP_PROOF_ROOT}/result_stage1/skipped.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/result_stage1/failed.jsonl}"

# ── vLLM ─────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/data/czx/models/Qwen3.5-9B}"
TP="${TP:-2}"
GPUS="${GPUS:-2,3}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOKEN_LIMIT="${TOKEN_LIMIT:-40960}"

# ── Batch / retry ─────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_RETRIES="${MAX_RETRIES:-3}"
INCLUDE_THINK_IN_DAG="${INCLUDE_THINK_IN_DAG:-1}"

THINK_FLAG="--include-think-in-dag"
case "${INCLUDE_THINK_IN_DAG,,}" in
  0|false|no|off)
    THINK_FLAG="--no-include-think-in-dag"
    ;;
esac

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
  --token-limit    "${TOKEN_LIMIT}" \
  --batch-size     "${BATCH_SIZE}" \
  --max-retries    "${MAX_RETRIES}" \
  "${THINK_FLAG}" \
  "$@"
