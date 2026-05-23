#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=path_env.sh
source "${SCRIPT_DIR}/path_env.sh"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON}"

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

append_configured_overrides() {
  local array_name="$1"
  shift
  local -n target="$1"
  if declare -p "${array_name}" >/dev/null 2>&1; then
    local -n source="${array_name}"
    target+=("${source[@]}")
  fi
}

split_cli_overrides() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      pipeline.*)
        PIPELINE_ARGS+=("${arg#pipeline.}")
        ;;
      rollout.*)
        ROLLOUT_ARGS+=("${arg#rollout.}")
        ;;
      step_proof.*)
        STEP_PROOF_ARGS+=("${arg#step_proof.}")
        ;;
      eval.*)
        EVAL_ARGS+=("${arg#eval.}")
        ;;
      *)
        PIPELINE_ARGS+=("${arg}")
        ;;
    esac
  done
}

run_pipeline() {
  local config_name="${1:-base}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  local config="${EXP_DIR}/configs/pipeline/${config_name}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Pipeline config not found: ${config}" >&2
    return 1
  fi

  PIPELINE_ARGS=()
  ROLLOUT_ARGS=()
  STEP_PROOF_ARGS=()
  EVAL_ARGS=()
  append_configured_overrides PIPELINE_OVERRIDES PIPELINE_ARGS
  append_configured_overrides ROLLOUT_OVERRIDES ROLLOUT_ARGS
  append_configured_overrides STEP_PROOF_OVERRIDES STEP_PROOF_ARGS
  append_configured_overrides EVAL_OVERRIDES EVAL_ARGS
  split_cli_overrides "$@"

  local pipeline_start
  pipeline_start="$(date +%s)"

  local -a stage_configs
  readarray -t stage_configs < <("${PYTHON_BIN}" - "${config}" "${PIPELINE_ARGS[@]}" <<'PY'
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
PY
)
  local idx
  for idx in "${!stage_configs[@]}"; do
    stage_configs[$idx]="${stage_configs[$idx]%$'\r'}"
  done

  echo "[timing][pipeline] start $(ts) config=${config_name} rollout_config=${stage_configs[0]} step_proof_config=${stage_configs[1]} eval_config=${stage_configs[2]}"

  if [[ "${RUN_ROLLOUT:-true}" == "true" ]]; then
    run_timed rollout bash "${SCRIPT_DIR}/01_rollout.sh" "${stage_configs[0]}" "${ROLLOUT_ARGS[@]}"
  else
    echo "[timing][rollout] skip $(ts)"
  fi

  if [[ "${RUN_STEP_PROOF:-true}" == "true" ]]; then
    run_timed step_proof bash "${SCRIPT_DIR}/02_step_proof.sh" "${stage_configs[1]}" "${STEP_PROOF_ARGS[@]}"
  else
    echo "[timing][step_proof] skip $(ts)"
  fi

  if [[ "${RUN_EVAL:-true}" == "true" ]]; then
    run_timed eval bash "${SCRIPT_DIR}/03_eval.sh" "${stage_configs[2]}" "${EVAL_ARGS[@]}"
  else
    echo "[timing][eval] skip $(ts)"
  fi

  local pipeline_end
  pipeline_end="$(date +%s)"
  echo "[timing][pipeline] done $(ts) elapsed=$(elapsed "$((pipeline_end - pipeline_start))")"
}
