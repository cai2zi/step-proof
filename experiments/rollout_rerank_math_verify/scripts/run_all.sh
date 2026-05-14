#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_NAME="${1:-default}"
CONFIG="${EXP_DIR}/configs/pipeline/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON:-${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}}"

readarray -t STAGE_CONFIGS < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["rollout_config"])
print(cfg["step_proof_config"])
print(cfg["eval_config"])
PY
)

bash "${SCRIPT_DIR}/01_rollout.sh" "${STAGE_CONFIGS[0]}"
bash "${SCRIPT_DIR}/02_step_proof.sh" "${STAGE_CONFIGS[1]}"
bash "${SCRIPT_DIR}/03_eval.sh" "${STAGE_CONFIGS[2]}"
