#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${EXP_DIR}/configs/experiment.yaml}"
LEAN4_PYTHON="${LEAN4_PYTHON:-/root/autodl-tmp/env/lean4/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/root/autodl-tmp/env/eval/bin/python}"

RUN_DIR="$("${LEAN4_PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(f"{cfg['output_root']}/{cfg['exp_name']}")
PY
)"

"${EVAL_PYTHON}" - <<'PY'
import importlib.util
missing = [m for m in ("math_verify",) if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing required module(s) in eval env: {missing}")
PY

for jsonl_path in "${RUN_DIR}"/math_verify/*.jsonl; do
  base="$(basename "${jsonl_path}" .jsonl)"
  case "${base}" in
    *_eval) continue ;;
  esac
  "${EVAL_PYTHON}" "${SCRIPT_DIR}/math_verify_eval.py" \
    --input-jsonl "${jsonl_path}" \
    --output-jsonl "${RUN_DIR}/math_verify/${base}_eval.jsonl"
done
