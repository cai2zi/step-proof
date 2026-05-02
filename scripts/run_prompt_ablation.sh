#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_PROOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEP_PROOF_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG_NAME="${CONFIG_NAME:-experiment_fdg}"
BASE_EXP_NAME="${BASE_EXP_NAME:-prompt_ablation_fdg}"

DEFAULT_FORMALIZER_PROMPT="formalize_obligation"
PAPER_FORMALIZER_PROMPT="formalize_obligation.paper_goedel_v2"
DEFAULT_PROVER_PROMPT="prove"
PAPER_PROVER_PROMPT="prove.paper_goedel_v2"

# Bit order: stage2 stage3.
# 11: stage2 paper prompt, stage3 paper prompt
# 01: stage2 default prompt, stage3 paper prompt
# 10: stage2 paper prompt, stage3 default prompt
CASES=("11" "01" "10")

for bits in "${CASES[@]}"; do
  stage2_bit="${bits:0:1}"
  stage3_bit="${bits:1:1}"

  formalizer_prompt="${DEFAULT_FORMALIZER_PROMPT}"
  prover_prompt="${DEFAULT_PROVER_PROMPT}"

  if [[ "${stage2_bit}" == "1" ]]; then
    formalizer_prompt="${PAPER_FORMALIZER_PROMPT}"
  fi
  if [[ "${stage3_bit}" == "1" ]]; then
    prover_prompt="${PAPER_PROVER_PROMPT}"
  fi

  exp_name="${BASE_EXP_NAME}_${bits}"

  echo "==> Running prompt ablation ${bits}"
  echo "    exp.name=${exp_name}"
  echo "    stage2.formalizer_prompt=${formalizer_prompt}"
  echo "    stage3.prover_prompt=${prover_prompt}"

  "${PYTHON}" "${STEP_PROOF_ROOT}/run_experiment.py" \
    --config-name "${CONFIG_NAME}" \
    "exp.name=${exp_name}" \
    "stage2.formalizer_prompt=${formalizer_prompt}" \
    "stage3.prover_prompt=${prover_prompt}" \
    "$@"
done
