#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG="${1:-${EXP_DIR}/configs/experiment.yaml}"
PYTHON="${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}"

EXP_NAME="$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["exp_name"])
PY
)"
OUTPUT_ROOT="$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["output_root"])
PY
)"

RUN_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
STEP_PROOF_INPUT_DIR="${RUN_DIR}/step_proof/input"
STEP_PROOF_RESULTS_ROOT="${RUN_DIR}/step_proof_results"
STEP_PROOF_EXP_NAME="fdg"

cd "${STEP_PROOF_ROOT}"
PYTHON="${PYTHON}" "${STEP_PROOF_ROOT}/scripts/run_experiment.sh" \
  --config-path "${EXP_DIR}/configs" \
  --config-name step_proof_fdg_rollout \
  "exp.root=${STEP_PROOF_RESULTS_ROOT}" \
  "exp.name=${STEP_PROOF_EXP_NAME}" \
  "stage1.parquet_dir=${STEP_PROOF_INPUT_DIR}" \
  "stage1.parquet_glob=rollout_flat.parquet"
