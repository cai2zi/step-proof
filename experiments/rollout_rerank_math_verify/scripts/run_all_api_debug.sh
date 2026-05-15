#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_NAME="${1:-API}"
if [[ $# -gt 0 ]]; then
  shift
fi
CONFIG="${EXP_DIR}/configs/pipeline/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON:-${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}}"

ts() {
  date +"%Y-%m-%d %H:%M:%S"
}

elapsed() {
  local seconds="$1"
  printf "%02d:%02d:%02d" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

run_timed() {
  local label="$1"
  shift
  local start
  local end
  start="$(date +%s)"
  echo "[timing][${label}] start $(ts)"
  "$@"
  end="$(date +%s)"
  echo "[timing][${label}] done $(ts) elapsed=$(elapsed "$((end - start))")"
}

PIPELINE_START="$(date +%s)"

readarray -t STAGE_CONFIGS < <("${PYTHON_BIN}" - "${CONFIG}" "$@" <<'PY'
import sys
import yaml

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

def set_dot_path(cfg, key, value):
    cur = cfg
    parts = [part for part in key.split(".") if part]
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    if parts:
        cur[parts[-1]] = value

if OmegaConf is None:
    cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
    for item in sys.argv[2:]:
        if "=" not in item:
            raise SystemExit(f"Config override must be key=value: {item!r}")
        key, value = item.split("=", 1)
        set_dot_path(cfg, key, yaml.safe_load(value))
else:
    cfg = OmegaConf.load(sys.argv[1])
    if len(sys.argv) > 2:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[2:]))
    cfg = OmegaConf.to_container(cfg, resolve=True)
print(cfg["rollout_config"])
print(cfg["step_proof_config"])
print(cfg["eval_config"])
print(cfg["rollout_name"])
print(cfg["step_proof_name"])
print(cfg.get("stage1_backend", "vllm"))
PY
)


export DASHSCOPE_API_KEY="sk-d4467529589744a390ca540fcb9f6013"


ROLLOUT_NAME="${STAGE_CONFIGS[3]}"
STEP_PROOF_NAME="${STAGE_CONFIGS[4]}"
STAGE1_BACKEND="${STAGE_CONFIGS[5]}"
echo "[timing][pipeline] start $(ts) config=${CONFIG_NAME} rollout=${ROLLOUT_NAME} step_proof=${STEP_PROOF_NAME} stage1_backend=${STAGE1_BACKEND}"
# run_timed rollout bash "${SCRIPT_DIR}/01_rollout.sh" "${STAGE_CONFIGS[0]}" \
#   "name=${ROLLOUT_NAME}"
run_timed step_proof bash "${SCRIPT_DIR}/02_step_proof.sh" "${STAGE_CONFIGS[1]}" \
  "rollout_name=${ROLLOUT_NAME}" \
  "name=${STEP_PROOF_NAME}" \
  "stage1.backend=${STAGE1_BACKEND}"
run_timed eval bash "${SCRIPT_DIR}/03_eval.sh" "${STAGE_CONFIGS[2]}" \
  "rollout_name=${ROLLOUT_NAME}" \
  "step_proof_name=${STEP_PROOF_NAME}"
PIPELINE_END="$(date +%s)"
echo "[timing][pipeline] done $(ts) elapsed=$(elapsed "$((PIPELINE_END - PIPELINE_START))")"