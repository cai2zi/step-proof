#!/usr/bin/env python3
"""Rebuild Stage3 prover prompts and report last-round token stats (plan: cot traces)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# step-proof/ on path for `proofflow`
_STEP_PROOF_ROOT = Path(__file__).resolve().parent.parent
if str(_STEP_PROOF_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEP_PROOF_ROOT))

from proofflow.prompt_builder import build_chat_messages

JsonDict = Dict[str, Any]
CZX_ROOT = Path(os.environ.get("CZX_ROOT", "/data/run01/scyb202/czx"))
WORK_ROOT = CZX_ROOT / "czx_work" / "step-proof"


def _split_lean_header_body(lean_code: str) -> Dict[str, str]:
    """Same rules as proofflow.fdg_stage_common._split_lean_header_body (no heavy imports)."""
    header_lines: List[str] = []
    body_lines: List[str] = []
    in_body = False
    for line in lean_code.splitlines():
        stripped = line.strip()
        if not in_body and (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
            or stripped == ""
        ):
            header_lines.append(line)
            continue
        in_body = True
        body_lines.append(line)
    return {
        "lean_header": "\n".join(header_lines).strip(),
        "lean_body": "\n".join(body_lines).strip(),
    }


def build_fdg_prove_messages(
    fact: JsonDict,
    *,
    prompt_name: str = "prove",
) -> List[Dict[str, str]]:
    """Mirror proofflow.fdg_stage_common.build_fdg_prove_messages."""
    proof_obligation = fact.get("proof_obligation") or {}
    lean_code = (fact.get("formalization") or {}).get("lean_code", "")
    messages = build_chat_messages(
        "prove",
        prompt_name=prompt_name,
        statement=str(proof_obligation.get("informal_statement_content", "")).strip(),
        lean_code=lean_code,
        **_split_lean_header_body(lean_code),
    )
    formalization = fact.get("formalization") or {}
    if formalization.get("lean_code") and not formalization.get("lean_pass"):
        messages[-1]["content"] += (
            "\n\nThe previous Lean4 code I sent you contains errors. Please take that into account."
        )
    return messages

DEFAULT_TOKENIZER = str(CZX_ROOT / "models" / "Goedel-Prover-V2-8B")
DEFAULT_PROVE_JSONL = str(WORK_ROOT / "results" / "qwen32B_1k" / "cot_traces" / "prove_last_attempts.jsonl")
DEFAULT_FORMAL_JSONL = str(WORK_ROOT / "results" / "qwen32B_1k" / "cot_traces" / "formal_last_attempts.jsonl")
DEFAULT_PROMPT_NAME = "prove.paper_goedel_v2"
RETRY_SUFFIX = "\n\n Based on these errors, please correct the previous response. "


def load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: JsonDict) -> Tuple[str, str]:
    return str(row.get("record_id", "")).strip(), str(row.get("fact_id", "")).strip()


def prove_category(row: JsonDict) -> Optional[str]:
    tries = int(row.get("tries") or 0)
    lv = bool(row.get("lean_verify"))
    if tries == 1:
        return "A"
    if tries == 2:
        return "B"
    if tries == 3 and lv:
        return "C"
    if tries == 3 and not lv:
        return "D"
    return None


def retry_user_content(entry: JsonDict) -> str:
    """Match FDGStage3Runner._validate_and_apply_one_output retry_error rules."""
    lean = str(entry.get("lean_code") or "").strip()
    err = entry.get("error_msg")
    err_s = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
    if lean:
        return f"Lean error/warnings: {err_s}"
    return f"Error: {err_s}"


def append_retry_turns(messages: List[Dict[str, str]], prove_row: JsonDict) -> None:
    """Append user turns from attempt_history top-level only (same order as runner)."""
    hist = prove_row.get("attempt_history") or []
    if not isinstance(hist, list):
        return
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        messages.append(
            {
                "role": "user",
                "content": retry_user_content(entry) + RETRY_SUFFIX,
            }
        )


def build_fact_from_join(prove_row: JsonDict, formal_row: JsonDict) -> JsonDict:
    return {
        "proof_obligation": prove_row.get("proof_obligation") or {},
        "formalization": {
            "lean_code": str(formal_row.get("lean_code") or ""),
            "lean_pass": bool(formal_row.get("lean_pass")),
        },
    }


def build_messages_last_round(
    prove_row: JsonDict,
    formal_row: JsonDict,
    *,
    prompt_name: str,
) -> List[Dict[str, str]]:
    fact = build_fact_from_join(prove_row, formal_row)
    messages = build_fdg_prove_messages(fact, prompt_name=prompt_name)
    append_retry_turns(messages, prove_row)
    return messages


def input_token_length(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    chat_template_kwargs: Dict[str, Any],
) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def output_token_length(tokenizer: Any, lean_code: str) -> int:
    return len(tokenizer.encode(lean_code or "", add_special_tokens=False))


def build_edges(max_val: int, *, start: int = 256) -> List[int]:
    edges = [0, start]
    while edges[-1] <= max_val:
        edges.append(edges[-1] * 2)
    return edges


def histogram_assign(edges: List[int], n: int) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return i
    return len(edges) - 1


def bucket_label(edges: List[int], idx: int) -> str:
    if idx < len(edges) - 1:
        return f"[{edges[idx]},{edges[idx + 1]})"
    return f"[{edges[-1]},inf)"


def histogram_for_values(values: List[int], *, start: int = 256) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "buckets": [],
        }
    mx = max(values)
    edges = build_edges(mx, start=start)
    counts: Dict[str, int] = {}
    for n in values:
        idx = histogram_assign(edges, n)
        label = bucket_label(edges, idx)
        counts[label] = counts.get(label, 0) + 1
    bucket_order = []
    for i in range(len(edges) - 1):
        bucket_order.append(bucket_label(edges, i))
    bucket_order.append(bucket_label(edges, len(edges) - 1))
    buckets_out = [{"range": lab, "count": counts.get(lab, 0)} for lab in bucket_order]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "buckets": buckets_out,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prove last-round token histograms (A–D).")
    p.add_argument("--prove-jsonl", type=Path, default=Path(DEFAULT_PROVE_JSONL))
    p.add_argument("--formal-jsonl", type=Path, default=Path(DEFAULT_FORMAL_JSONL))
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--prompt-name",
        type=str,
        default=DEFAULT_PROMPT_NAME,
        help="Stem under prompts/{system,user}/ e.g. prove.paper_goedel_v2",
    )
    p.add_argument(
        "--chat-template-kwargs-json",
        type=str,
        default='{"enable_thinking": false}',
        help="JSON object passed to apply_chat_template",
    )
    p.add_argument("--bucket-start", type=int, default=256)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chat_template_kwargs: Dict[str, Any] = json.loads(args.chat_template_kwargs_json)
    if not isinstance(chat_template_kwargs, dict):
        raise SystemExit("--chat-template-kwargs-json must be a JSON object")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    prove_rows = load_jsonl(args.prove_jsonl)
    formal_rows = load_jsonl(args.formal_jsonl)
    formal_map: Dict[Tuple[str, str], JsonDict] = {}
    for fr in formal_rows:
        formal_map[row_key(fr)] = fr

    excluded: Dict[str, int] = {}
    unmatched = 0
    mismatch_hist_len = 0

    # category -> metric -> list of token lengths
    series: Dict[str, Dict[str, List[int]]] = {
        cat: {"input": [], "output": [], "sum": []}
        for cat in ("A", "B", "C", "D")
    }

    for pr in prove_rows:
        cat = prove_category(pr)
        if cat is None:
            key = f"tries_{pr.get('tries')}"
            excluded[key] = excluded.get(key, 0) + 1
            continue

        fk = row_key(pr)
        formal = formal_map.get(fk)
        if formal is None:
            unmatched += 1
            continue

        tries = int(pr.get("tries") or 0)
        hist = pr.get("attempt_history") or []
        top_hist = [x for x in hist if isinstance(x, dict)] if isinstance(hist, list) else []
        if len(top_hist) != max(0, tries - 1):
            mismatch_hist_len += 1

        messages = build_messages_last_round(pr, formal, prompt_name=args.prompt_name)
        inp = input_token_length(tokenizer, messages, chat_template_kwargs)
        out = output_token_length(tokenizer, str(pr.get("lean_code") or ""))
        s = inp + out

        series[cat]["input"].append(inp)
        series[cat]["output"].append(out)
        series[cat]["sum"].append(s)

    report: Dict[str, Any] = {
        "prove_jsonl": str(args.prove_jsonl),
        "formal_jsonl": str(args.formal_jsonl),
        "tokenizer": args.tokenizer,
        "prompt_name": args.prompt_name,
        "excluded_breakdown": excluded,
        "eligible_ABCD": {k: len(series[k]["input"]) for k in "ABCD"},
        "unmatched_formal_rows": unmatched,
        "attempt_history_len_mismatch_with_tries": mismatch_hist_len,
        "categories": {},
    }

    for cat in "ABCD":
        report["categories"][cat] = {
            "n": len(series[cat]["input"]),
            "input_tokens": histogram_for_values(series[cat]["input"], start=args.bucket_start),
            "output_tokens": histogram_for_values(series[cat]["output"], start=args.bucket_start),
            "sum_tokens": histogram_for_values(series[cat]["sum"], start=args.bucket_start),
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
