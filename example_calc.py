"""
计算题流水线示例：从 data_check 生成的 JSONL（每行 question + response）读取，
将 question 作为 Problem、response 作为 Raw CoT，走 task_profile=\"calc\" 的 DAG + 形式化 + 证明。
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from proofflow import LeanServer, LLMManager, ProofFlow

load_dotenv()

# 默认 JSONL：与 data_check/check_p.py 生成的 question_response_10.jsonl 对齐
DEFAULT_JSONL = Path(__file__).resolve().parent.parent / "data_check" / "question_response_10.jsonl"

GRAPH_BASE_URL = os.getenv("GRAPH_BASE_URL", "http://127.0.0.1:8001/v1")
FORMALIZER_BASE_URL = os.getenv("FORMALIZER_BASE_URL", "http://127.0.0.1:8002/v1")
PROVER_BASE_URL = os.getenv("PROVER_BASE_URL", "http://127.0.0.1:8003/v1")
DUMMY_API_KEY = os.getenv("DUMMY_API_KEY", "dummy")
MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")


def load_jsonl_records(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Run calc ProofFlow on JSONL (question, response).")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="JSONL path (each line: {\"question\": ..., \"response\": ...})",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Only run this 0-based line index (-1 = run all lines)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "calc_runs",
        help="Directory for per-record outputs",
    )
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"JSONL not found: {args.jsonl}. Run data_check/check_p.py first.")

    records = load_jsonl_records(args.jsonl)
    if not records:
        raise SystemExit("JSONL is empty.")

    lean_server = LeanServer(project_path=MATHLIB_PATH)

    graph_model = LLMManager(
        model_info={
            "api_key": DUMMY_API_KEY,
            "base_url": GRAPH_BASE_URL,
            "model": os.getenv("GRAPH_MODEL", "qwen3.5-9b"),
        },
    )
    formalize_model = LLMManager(
        model_info={
            "api_key": DUMMY_API_KEY,
            "base_url": FORMALIZER_BASE_URL,
            "model": os.getenv("FORMALIZER_MODEL", "goedel-formalizer-v2-8b"),
        },
    )
    solver_model = LLMManager(
        model_info={
            "api_key": DUMMY_API_KEY,
            "base_url": PROVER_BASE_URL,
            "model": os.getenv("PROVER_MODEL", "goedel-prover-v2-8b"),
        },
    )

    if args.index < 0:
        indices = list(range(len(records)))
    else:
        if args.index >= len(records):
            raise SystemExit(f"--index {args.index} out of range (0..{len(records)-1})")
        indices = [args.index]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        rec = records[i]
        if "question" not in rec or "response" not in rec:
            raise KeyError(f"Line {i}: expected keys question and response, got {list(rec.keys())}")

        problem = rec["question"]
        raw_cot = rec["response"]

        print(f"\n========== Record {i} (of {len(records)} lines) ==========\n")
        proof_flow = ProofFlow(
            lean_server=lean_server,
            graph_model_manager=graph_model,
            formalize_model_manager=formalize_model,
            solver_model_manager=solver_model,
            score_model_manager=None,
            verbose=True,
            task_profile="calc",
        )
        proof_flow.autoformalize_series(problem=problem, raw_cot=raw_cot)

        prefix = args.out_dir / f"record_{i}"
        lean_path = prefix.with_suffix(".lean.txt")
        lean_path.write_text(proof_flow.get_lean_code(), encoding="utf-8")
        summary = proof_flow.summary()
        (prefix.with_suffix(".summary.json")).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        proof_flow.plot_dag(str(prefix) + "_dag.png")
        proof_flow.interactive_dag(str(prefix) + "_dag.html")
        print(f"Saved: {lean_path}, summary, {prefix}_dag.png/html")


if __name__ == "__main__":
    main()
