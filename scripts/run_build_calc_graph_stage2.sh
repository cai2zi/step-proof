#!/usr/bin/env bash
set -euo pipefail

# 阶段二 form + prove（本地 vLLM + 并发 Lean 校验）。
# 已内置默认参数，可直接执行；也可追加命令行参数覆盖，或通过环境变量覆盖。
#
# 示例:
#   ./scripts/run_build_calc_graph_stage2.sh
#   ./scripts/run_build_calc_graph_stage2.sh --limit 10
#   LIMIT=10 ./scripts/run_build_calc_graph_stage2.sh
#
# 指定 Python（可选）:
#   PYTHON=/opt/anaconda3/envs/lean4-czx/bin/python ./scripts/run_build_calc_graph_stage2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-/opt/anaconda3/envs/lean4-czx/bin/python}"

# ── 输入 / 输出 ───────────────────────────────────────────────────────────
INFILE="${INFILE:-${STEP_PROOF_ROOT}/calc_runs/graphs.jsonl}"
OUT_JSONL="${OUT_JSONL:-${STEP_PROOF_ROOT}/calc_runs/stage2_results.jsonl}"
FAILED_JSONL="${FAILED_JSONL:-${STEP_PROOF_ROOT}/calc_runs/stage2_failed.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STEP_PROOF_ROOT}/calc_runs/stage2_ckpt}"
LIMIT="${LIMIT:--1}"

# ── Lean 校验 ────────────────────────────────────────────────────────────
MATHLIB_PATH="${MATHLIB_PATH:-/data/czx/mathlib4}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-16}"
LEAN_TEMP_DIR="${LEAN_TEMP_DIR:-${STEP_PROOF_ROOT}/calc_runs/lean_jobs}"

# ── 运行时 / 调度 ────────────────────────────────────────────────────────
GPUS="${GPUS:-4,5,6,7}"
FORMALIZER_GPUS="${FORMALIZER_GPUS:-4,5}"
PROVER_GPUS="${PROVER_GPUS:-6,7}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ID_SCHEMA_MODE="${ID_SCHEMA_MODE:-calc}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-200}"

# ── Formalizer ───────────────────────────────────────────────────────────
FORMALIZER_MODEL_PATH="${FORMALIZER_MODEL_PATH:-/data/czx/models/Goedel-Formalizer-V2-8B}"
FORMALIZER_TP="${FORMALIZER_TP:-2}"
FORMALIZER_MAX_TOKENS="${FORMALIZER_MAX_TOKENS:-8192}"
FORMALIZER_TOKEN_LIMIT="${FORMALIZER_TOKEN_LIMIT:-32768}"
FORMALIZER_TEMPERATURE="${FORMALIZER_TEMPERATURE:-0.2}"
FORMALIZER_RETRIES="${FORMALIZER_RETRIES:-3}"
FORM_BATCH_SIZE="${FORM_BATCH_SIZE:-64}"

# ── Prover ───────────────────────────────────────────────────────────────
PROVER_MODEL_PATH="${PROVER_MODEL_PATH:-/data/czx/models/Goedel-Prover-V2-8B}"
PROVER_TP="${PROVER_TP:-2}"
PROVER_MAX_TOKENS="${PROVER_MAX_TOKENS:-8192}"
PROVER_TOKEN_LIMIT="${PROVER_TOKEN_LIMIT:-32768}"
PROVER_TEMPERATURE="${PROVER_TEMPERATURE:-0.2}"
PROVER_RETRIES="${PROVER_RETRIES:-3}"
PROVE_BATCH_SIZE="${PROVE_BATCH_SIZE:-64}"

exec "${PYTHON}" "${STEP_PROOF_ROOT}/build_calc_graph_stage2.py" \
  --infile "${INFILE}" \
  --out "${OUT_JSONL}" \
  --failed "${FAILED_JSONL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --limit "${LIMIT}" \
  --mathlib-path "${MATHLIB_PATH}" \
  --lean-check-concurrency "${LEAN_CHECK_CONCURRENCY}" \
  --lean-temp-dir "${LEAN_TEMP_DIR}" \
  --gpus "${GPUS}" \
  --formalizer-gpus "${FORMALIZER_GPUS}" \
  --prover-gpus "${PROVER_GPUS}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --id-schema-mode "${ID_SCHEMA_MODE}" \
  --batch-wait-ms "${BATCH_WAIT_MS}" \
  --formalizer-model-path "${FORMALIZER_MODEL_PATH}" \
  --formalizer-tensor-parallel-size "${FORMALIZER_TP}" \
  --formalizer-max-tokens "${FORMALIZER_MAX_TOKENS}" \
  --formalizer-token-limit "${FORMALIZER_TOKEN_LIMIT}" \
  --formalizer-temperature "${FORMALIZER_TEMPERATURE}" \
  --formalizer-retries "${FORMALIZER_RETRIES}" \
  --form-batch-size "${FORM_BATCH_SIZE}" \
  --prover-model-path "${PROVER_MODEL_PATH}" \
  --prover-tensor-parallel-size "${PROVER_TP}" \
  --prover-max-tokens "${PROVER_MAX_TOKENS}" \
  --prover-token-limit "${PROVER_TOKEN_LIMIT}" \
  --prover-temperature "${PROVER_TEMPERATURE}" \
  --prover-retries "${PROVER_RETRIES}" \
  --prove-batch-size "${PROVE_BATCH_SIZE}" \
  "$@"
