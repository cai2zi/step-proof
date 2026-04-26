#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-python}"
EXP_ROOT="${EXP_ROOT:-results}"

copy_stage1_from_source() {
  local src_exp_name="$1"
  local dst_exp_name="$2"

  local src_stage1_dir="${STEP_PROOF_ROOT}/${EXP_ROOT}/${src_exp_name}/result_stage1"
  local dst_exp_dir="${STEP_PROOF_ROOT}/${EXP_ROOT}/${dst_exp_name}"
  local dst_stage1_dir="${dst_exp_dir}/result_stage1"

  if [[ ! -d "${src_stage1_dir}" ]]; then
    echo "[ctx] source stage1 dir not found: ${src_stage1_dir}" >&2
    exit 1
  fi

  mkdir -p "${dst_exp_dir}"
  rm -rf "${dst_stage1_dir}"
  cp -a "${src_stage1_dir}" "${dst_stage1_dir}"
  echo "[ctx] copied stage1: ${src_exp_name} -> ${dst_exp_name}"
}

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
run_one "debug_exp" false false "$@"
copy_stage1_from_source "debug_exp" "ctx01"
run_one "ctx01" false true "$@"
copy_stage1_from_source "debug_exp" "ctx10"
run_one "ctx10" true false "$@"
copy_stage1_from_source "debug_exp" "ctx11"
run_one "ctx11" true true "$@"
