#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_pipeline_lib.sh
source "${SCRIPT_DIR}/run_pipeline_lib.sh"

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
  echo "Usage: $0 <variant-name-or-yaml> [pipeline./rollout./step_proof./eval. overrides...]" >&2
  exit 2
fi
shift || true

if [[ "${VARIANT}" == */* || "${VARIANT}" == *.yaml ]]; then
  VARIANT_PATH="${VARIANT}"
else
  VARIANT_PATH="${EXP_DIR}/configs/variants/${VARIANT}.yaml"
fi

if [[ ! -f "${VARIANT_PATH}" ]]; then
  echo "Variant config not found: ${VARIANT_PATH}" >&2
  exit 1
fi

eval "$("${PYTHON_BIN}" - "${VARIANT_PATH}" <<'PY'
import shlex
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}

def normalize_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

def q(value):
    return shlex.quote(str(value))

def emit_scalar(name, default):
    print(f"{name}={q(normalize_scalar(cfg.get(name.lower(), default)))}")

def emit_array(name, key):
    values = cfg.get(key, []) or []
    joined = " ".join(q(item) for item in values)
    print(f"{name}=({joined})")

emit_scalar("CONFIG_NAME", "base")
emit_scalar("RUN_ROLLOUT", "true")
emit_scalar("RUN_STEP_PROOF", "true")
emit_scalar("RUN_EVAL", "true")
emit_array("PIPELINE_OVERRIDES", "pipeline_overrides")
emit_array("ROLLOUT_OVERRIDES", "rollout_overrides")
emit_array("STEP_PROOF_OVERRIDES", "step_proof_overrides")
emit_array("EVAL_OVERRIDES", "eval_overrides")
PY
)"

run_pipeline "${CONFIG_NAME}" "$@"
