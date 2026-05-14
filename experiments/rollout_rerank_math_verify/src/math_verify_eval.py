#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer/gold JSONL using Math-Verify without pandas.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def _gold_for_verify(gold: str) -> str:
    text = str(gold).strip()
    if "$" in text:
        return text
    if any(token in text for token in ("\\", "{", "}", "^", "_")):
        return f"${text}$"
    return text


def main() -> None:
    args = parse_args()
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6,
    )

    with args.input_jsonl.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"empty input jsonl: {args.input_jsonl}")

    out_rows: List[Dict[str, Any]] = []
    correct = 0
    for row in rows:
        grade = 0
        extracted_gold = ""
        extracted_answer = ""
        error = ""
        try:
            grade, extracted = verify_func([_gold_for_verify(row["gold"])], [row["answer"]])
            if extracted:
                extracted_gold = str(extracted[0])
                extracted_answer = str(extracted[1])
        except Exception as exc:
            error = str(exc)
        is_correct = int(grade == 1)
        correct += is_correct
        out_rows.append(
            {
                **row,
                "extracted_gold": extracted_gold,
                "extracted_answer": extracted_answer,
                "is_correct": str(is_correct),
                "error": error,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(out_rows)
    print(f"input={args.input_jsonl}")
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={correct / total:.6f}")
    print(f"output={args.output_jsonl}")


if __name__ == "__main__":
    main()
