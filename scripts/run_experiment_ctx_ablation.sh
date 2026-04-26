#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-python}"

run_one() {
  local exp_name="$1"
  local include_parent_statement="$2"
  local include_parent_nl="$3"
  shift 3
  echo "[ctx] exp.name=${exp_name} include_parent_statement=${include_parent_statement} include_parent_nl=${include_parent_nl}"

  "${PYTHON}" "${STEP_PROOF_ROOT}/run_experiment.py" \
    "exp.name=${exp_name}" \
    "stage2.include_parent_statement=${include_parent_statement}" \
    "stage2.include_parent_nl=${include_parent_nl}" \
    "$@"
}

run_one "ctx01" false true "$@"
run_one "ctx10" true false "$@"
run_one "ctx11" true true "$@"