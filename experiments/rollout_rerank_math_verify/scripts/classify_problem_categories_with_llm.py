#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

try:
    from path_defaults import step_proofs_root
except Exception:  # pragma: no cover
    step_proofs_root = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

CATEGORIES = ["Number Theory", "Discrete & Prob", "Calculus", "Geometry", "Algebra"]
DEFAULT_API_BASE_URL = "https://poloapi.top/v1"
DEFAULT_API_KEY_ENV = "POLOAI_API_KEY"
DEFAULT_MODEL = "gpt-5-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify unique step-proof problems into broad math categories with an LLM."
    )
    parser.add_argument(
        "project_root",
        type=Path,
        help="Project root, e.g. D:/program/research or D:/program/research/step-proof.",
    )
    parser.add_argument("exp_name", help="Experiment name, with or without step_proof_ prefix.")
    parser.add_argument(
        "--step-proof-root",
        type=Path,
        default=None,
        help="Directory containing step_proof_* experiments. If omitted, inferred from project_root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Default: <exp_dir>/analysis/problem_categories.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max problems to classify. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=20, help="Problems per API call. Default: 20.")
    parser.add_argument("--resume", action="store_true", help="Skip parent_ids already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt batches without calling the API.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key", default="", help="API key passed directly on the command line.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    return parser.parse_args()


def normalize_exp_name(name: str) -> str:
    name = str(name).strip()
    return name if name.startswith("step_proof_") else f"step_proof_{name}"


def infer_step_proof_root(project_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    project_root = project_root.expanduser().resolve()
    candidates = [
        project_root / "czx_work" / "step-proof" / "rollout_rerank_math_verify" / "outputs" / "step_proofs",
        project_root.parent / "czx_work" / "step-proof" / "rollout_rerank_math_verify" / "outputs" / "step_proofs",
    ]
    if step_proofs_root is not None:
        try:
            candidates.append(step_proofs_root())
        except Exception:
            pass
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def exp_dir_from_name(step_proof_root: Path, exp_name: str) -> Path:
    path = step_proof_root / normalize_exp_name(exp_name)
    if not path.is_dir():
        raise SystemExit(f"experiment not found: {path}")
    return path


def read_jsonl(path: Path) -> Iterable[JsonDict]:
    if not path.is_file():
        raise SystemExit(f"required file not found: {path}")
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def record_id(record: JsonDict) -> str:
    return str((record.get("meta") or {}).get("record_id") or "").strip()


def parent_id_from_record_id(rid: str) -> str:
    return rid.rsplit("__rollout_", 1)[0] if "__rollout_" in rid else rid


def collect_problems(stage3_path: Path) -> list[JsonDict]:
    by_parent: dict[str, JsonDict] = {}
    for rec in read_jsonl(stage3_path):
        rid = record_id(rec)
        parent_id = parent_id_from_record_id(rid)
        if not parent_id or parent_id in by_parent:
            continue
        input_payload = rec.get("input") or {}
        problem = str(input_payload.get("problem") or "").strip()
        if not problem:
            continue
        by_parent[parent_id] = {
            "parent_id": parent_id,
            "example_record_id": rid,
            "problem": problem,
        }
    return list(by_parent.values())


def load_done_parent_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        parent_id = str(row.get("parent_id") or "").strip()
        if parent_id:
            done.add(parent_id)
    return done


def batches(rows: list[JsonDict], size: int) -> Iterable[list[JsonDict]]:
    size = max(1, size)
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def build_messages(rows: list[JsonDict]) -> list[JsonDict]:
    system = (
        "You classify math competition problems into exactly one category. "
        "Allowed categories: Number Theory, Discrete & Prob, Calculus, Geometry, Algebra. "
        "Return only valid JSON."
    )
    items = [
        {
            "parent_id": row["parent_id"],
            "problem": row["problem"],
        }
        for row in rows
    ]
    user = (
        "Classify each problem into exactly one of these categories:\n"
        "Number Theory, Discrete & Prob, Calculus, Geometry, Algebra.\n\n"
        "Use Discrete & Prob for combinatorics, counting, probability, graph theory, recurrence, "
        "or finite discrete structures. Use Algebra for equations, inequalities, functions, "
        "polynomials, sequences unless the main method is clearly another category.\n\n"
        "Return JSON with this shape only:\n"
        '{"items":[{"parent_id":"...","category":"...","reason":"short reason"}]}\n\n'
        f"Problems:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_chat_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[JsonDict],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> JsonDict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(max(0, retries) + 1):
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            return {
                "ok": True,
                "content": content,
                "raw_response": data,
                "duration_seconds": time.time() - started,
                "attempt": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(max(0.0, retry_sleep))
    return {
        "ok": False,
        "content": "",
        "error": last_error,
        "attempt": max(0, retries) + 1,
    }


def extract_json_object(text: str) -> JsonDict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def normalize_category(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "discrete": "Discrete & Prob",
        "discrete math": "Discrete & Prob",
        "combinatorics": "Discrete & Prob",
        "probability": "Discrete & Prob",
        "discrete & probability": "Discrete & Prob",
        "number theory": "Number Theory",
        "geometry": "Geometry",
        "algebra": "Algebra",
        "calculus": "Calculus",
    }
    if text in CATEGORIES:
        return text
    return aliases.get(text.lower(), "")


def rows_from_response(rows: list[JsonDict], result: JsonDict) -> list[JsonDict]:
    content = str(result.get("content") or "")
    parsed = extract_json_object(content)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    item_by_parent: dict[str, JsonDict] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            parent_id = str(item.get("parent_id") or "").strip()
            if parent_id:
                item_by_parent[parent_id] = item

    out: list[JsonDict] = []
    for row in rows:
        item = item_by_parent.get(row["parent_id"], {})
        category = normalize_category(item.get("category"))
        out.append(
            {
                "parent_id": row["parent_id"],
                "example_record_id": row["example_record_id"],
                "problem": row["problem"],
                "category": category or "UNKNOWN",
                "reason": str(item.get("reason") or "").strip(),
                "ok": bool(result.get("ok")) and bool(category),
                "raw_category": item.get("category", ""),
                "raw_response": content if not category else "",
            }
        )
    return out


def append_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    step_proof_root = infer_step_proof_root(args.project_root, args.step_proof_root)
    exp_dir = exp_dir_from_name(step_proof_root, args.exp_name)
    stage3_path = exp_dir / "step_proof_results" / "result_stage3" / "stage3_results.jsonl"
    output = args.output or (exp_dir / "analysis" / "problem_categories.jsonl")

    problems = collect_problems(stage3_path)
    if args.resume:
        done = load_done_parent_ids(output)
        problems = [row for row in problems if row["parent_id"] not in done]
    if args.limit > 0:
        problems = problems[: args.limit]
    if not problems:
        print("No problems to classify.")
        return

    api_key = ""
    if not args.dry_run:
        api_key = str(args.api_key or "").strip() or os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                "API key is missing. Pass --api-key directly, or set the environment "
                f"variable named by --api-key-env ({args.api_key_env})."
            )

    total_written = 0
    for idx, batch in enumerate(batches(problems, args.batch_size), 1):
        messages = build_messages(batch)
        print(f"[batch {idx}] classify {len(batch)} problems")
        if args.dry_run:
            out_rows = [
                {
                    **row,
                    "category": "DRY_RUN",
                    "reason": "",
                    "ok": True,
                    "messages": messages,
                }
                for row in batch
            ]
        else:
            result = call_chat_api(
                base_url=args.api_base_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            out_rows = rows_from_response(batch, result)
        append_jsonl(output, out_rows)
        total_written += len(out_rows)

    print(f"Wrote {total_written} row(s) to {output}")


if __name__ == "__main__":
    main()
