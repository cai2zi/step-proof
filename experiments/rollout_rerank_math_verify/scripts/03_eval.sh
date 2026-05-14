#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEP_PROOF_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-math_verify}"
if [[ $# -gt 0 ]]; then
  shift
fi
CONFIG="${EXP_DIR}/configs/eval/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON:-${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}}"

readarray -t PATHS < <("${PYTHON_BIN}" - "${CONFIG}" "$@" <<'PY'
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
root = cfg["output_root"]
step_proof_name = cfg["step_proof_name"]
eval_python = (cfg.get("env") or {}).get("eval_python", "/root/autodl-tmp/env/eval/bin/python")
print(f"{root}/step_proofs/step_proof_{step_proof_name}/math_verify")
print(eval_python)
PY
)

MATH_VERIFY_DIR="${PATHS[0]}"
EVAL_PYTHON_BIN="${EVAL_PYTHON:-${PATHS[1]}}"

cd "${STEP_PROOF_ROOT}"
"${PYTHON_BIN}" "${EXP_DIR}/src/score_step_proof.py" --config "${CONFIG}" "$@"
"${PYTHON_BIN}" "${EXP_DIR}/src/build_math_verify_inputs.py" --config "${CONFIG}" "$@"

"${EVAL_PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [m for m in ("math_verify",) if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing required module(s) in eval env: {missing}")
PY

shopt -s nullglob
for jsonl_path in "${MATH_VERIFY_DIR}"/*.jsonl; do
  base="$(basename "${jsonl_path}" .jsonl)"
  case "${base}" in
    *_eval) continue ;;
  esac
  "${EVAL_PYTHON_BIN}" "${EXP_DIR}/src/math_verify_eval.py" \
    --input-jsonl "${jsonl_path}" \
    --output-jsonl "${MATH_VERIFY_DIR}/${base}_eval.jsonl"
done

"${PYTHON_BIN}" "${EXP_DIR}/src/summarize_results.py" --config "${CONFIG}" "$@"
