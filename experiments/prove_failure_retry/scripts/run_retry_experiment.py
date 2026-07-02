#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None  # type: ignore[assignment]


JsonDict = Dict[str, Any]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
STEP_PROOF_ROOT = EXPERIMENT_DIR.parents[1]
PROMPT_ROOT = STEP_PROOF_ROOT / "prompts"
if str(STEP_PROOF_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_PROOF_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and retry prover-caused step-proof failures.")
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def _set_dot_path(cfg: JsonDict, key: str, value: Any) -> None:
    cur = cfg
    parts = [part for part in key.split(".") if part]
    for part in parts[:-1]:
        next_value = cur.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[part] = next_value
        cur = next_value
    if parts:
        cur[parts[-1]] = value


def _select_dot_path(cfg: JsonDict, key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _resolve_env_expr(expr: str) -> str:
    payload = expr[len("oc.env:") :]
    if "," in payload:
        name, default = payload.split(",", 1)
    else:
        name, default = payload, ""
    return os.getenv(name, default)


def _resolve_fallback(value: Any, root: JsonDict) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_fallback(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_fallback(item, root) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        if expr.startswith("oc.env:"):
            return _resolve_env_expr(expr)
        try:
            return str(_select_dot_path(root, expr))
        except KeyError:
            return match.group(0)

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def load_config(path: Path, overrides: List[str]) -> JsonDict:
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Config override must be key=value: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dot_path(cfg, key, yaml.safe_load(raw_value))
    return _resolve_fallback(cfg, cfg)


def read_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_unclosed_think_block(text: Any) -> bool:
    value = "" if text is None else str(text)
    without_closed = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return bool(re.search(r"<think\b[^>]*>.*\Z", without_closed, flags=re.DOTALL | re.IGNORECASE))


def split_rollout_id(record_id: str) -> Tuple[str, int]:
    marker = "__rollout_"
    if marker not in record_id:
        raise ValueError(f"not a rollout record id: {record_id}")
    parent_id, rollout = record_id.rsplit(marker, 1)
    return parent_id, int(rollout)


def record_id_of(record: JsonDict) -> str:
    rid = str((record.get("meta") or {}).get("record_id") or record.get("id") or "").strip()
    if not rid:
        raise ValueError("stage3 record is missing meta.record_id")
    return rid


def output_root(cfg: JsonDict) -> Path:
    return Path(str(cfg["output_root"])) / str(cfg["exp_name"])


def source_exp_dir(cfg: JsonDict) -> Path:
    return Path(str((cfg.get("source") or {})["exp_dir"]))


def source_path(cfg: JsonDict, key: str) -> Path:
    source = cfg.get("source") or {}
    return source_exp_dir(cfg) / str(source[key])


def node_key(record_id: str, fact_id: str) -> str:
    return f"{record_id}::{fact_id}"


def load_source_records(cfg: JsonDict) -> Tuple[List[JsonDict], Dict[str, JsonDict]]:
    stage3_path = source_path(cfg, "stage3_jsonl")
    math_path = source_path(cfg, "math_verify_all_rollouts_eval")
    records = list(read_jsonl(stage3_path))
    math_rows = {str(row.get("id") or ""): row for row in read_jsonl(math_path)}
    return records, math_rows


def select_subset(cfg: JsonDict, records: List[JsonDict], math_rows: Dict[str, JsonDict]) -> Tuple[List[JsonDict], JsonDict]:
    by_parent: Dict[str, List[JsonDict]] = {}
    record_ids_without_math: List[str] = []
    rejected_unclosed: List[str] = []

    for record in records:
        rid = record_id_of(record)
        parent_id, _rollout_id = split_rollout_id(rid)
        by_parent.setdefault(parent_id, []).append(record)

    eligible_parent_ids: List[str] = []
    reject_think = bool(((cfg.get("filter") or {}).get("reject_unclosed_think_problem", True)))
    for parent_id in sorted(by_parent):
        reject = False
        for record in by_parent[parent_id]:
            rid = record_id_of(record)
            row = math_rows.get(rid)
            raw_cot = (record.get("input") or {}).get("raw_cot")
            if reject_think and has_unclosed_think_block(raw_cot):
                reject = True
            if row is None:
                record_ids_without_math.append(rid)
                continue
            if reject_think and has_unclosed_think_block(row.get("answer")):
                reject = True
        if reject:
            rejected_unclosed.append(parent_id)
        else:
            eligible_parent_ids.append(parent_id)

    sample_n = int((cfg.get("run") or {}).get("sample_problems", -1))
    seed = int((cfg.get("run") or {}).get("seed", 0))
    if sample_n < 0 or sample_n >= len(eligible_parent_ids):
        selected_parent_ids = eligible_parent_ids
    else:
        rng = random.Random(seed)
        selected_parent_ids = sorted(rng.sample(eligible_parent_ids, sample_n))

    selected_records: List[JsonDict] = []
    for parent_id in selected_parent_ids:
        selected_records.extend(sorted(by_parent[parent_id], key=lambda rec: split_rollout_id(record_id_of(rec))[1]))

    meta = {
        "total_records": len(records),
        "total_problems": len(by_parent),
        "eligible_problems_after_think_filter": len(eligible_parent_ids),
        "selected_problems": len(selected_parent_ids),
        "selected_records": len(selected_records),
        "rejected_unclosed_think_problem_count": len(rejected_unclosed),
        "rejected_unclosed_think_problem_ids": rejected_unclosed[:50],
        "record_ids_without_math_verify": record_ids_without_math[:50],
        "record_ids_without_math_verify_count": len(record_ids_without_math),
    }
    return selected_records, meta


def fact_origin(fact: JsonDict) -> str:
    return str(fact.get("origin") or "").strip().lower()


def is_skipped_derived(fact: JsonDict) -> bool:
    return fact_origin(fact) == "derived" and int(fact.get("skip", 0) or 0) == 1


def is_required_fact(fact: JsonDict, include_skipped_derived: bool) -> bool:
    if fact_origin(fact) not in {"derived", "answer"}:
        return False
    return include_skipped_derived or not is_skipped_derived(fact)


def fact_form_verified(fact: JsonDict) -> bool:
    formal = fact.get("formalization") or {}
    return bool(formal.get("lean_pass"))


def fact_prove_verified(fact: JsonDict) -> bool:
    solved = fact.get("solved_lemma") or {}
    return bool(solved.get("lean_verify"))


def bucket(percent: float, step: int = 10) -> str:
    clipped = max(0.0, min(100.0, percent))
    if clipped >= 100.0:
        return "100%"
    left = int(clipped // step) * step
    return f"{left}-{min(left + step, 100)}%"


def empty_hist(step: int = 10) -> Dict[str, int]:
    return {f"{left}-{min(left + step, 100)}%": 0 for left in range(0, 100, step)} | {"100%": 0}


def compute_stage3_stats(records: List[JsonDict], top_n: int = 10) -> JsonDict:
    def one(include_skipped: bool) -> JsonDict:
        prove_hist = empty_hist()
        form_hist = empty_hist()
        top_ids: Dict[str, List[str]] = {key: [] for key in prove_hist}
        global_nodes_total = 0
        prove_required_total = 0
        form_required_total = 0
        prove_verified = 0
        form_verified = 0
        all_prove_verified_records = 0
        final_answer_wrong_records = 0

        for record in records:
            rid = record_id_of(record)
            facts = ((record.get("results") or {}).get("facts") or [])
            facts = [fact for fact in facts if isinstance(fact, dict)]
            global_nodes_total += len(facts)
            required = [fact for fact in facts if is_required_fact(fact, include_skipped)]
            prove_required = len(required)
            form_required = len(required)
            rec_prove_verified = sum(1 for fact in required if fact_prove_verified(fact))
            rec_form_verified = sum(1 for fact in required if fact_form_verified(fact))
            prove_required_total += prove_required
            form_required_total += form_required
            prove_verified += rec_prove_verified
            form_verified += rec_form_verified
            if prove_required and rec_prove_verified == prove_required:
                all_prove_verified_records += 1
            finals = [fact for fact in required if fact_origin(fact) == "answer" or bool(fact.get("is_final_answer"))]
            if finals and not any(fact_prove_verified(fact) for fact in finals):
                final_answer_wrong_records += 1

            prove_pct = (rec_prove_verified / prove_required * 100.0) if prove_required else 0.0
            form_pct = (rec_form_verified / form_required * 100.0) if form_required else 0.0
            prove_bucket = bucket(prove_pct)
            form_bucket = bucket(form_pct)
            prove_hist[prove_bucket] += 1
            form_hist[form_bucket] += 1
            if len(top_ids[prove_bucket]) < top_n:
                top_ids[prove_bucket].append(rid)

        total = len(records)
        return {
            "all_nodes_prove_verified_records": all_prove_verified_records,
            "all_nodes_prove_verified_records_ratio": (
                all_prove_verified_records / total * 100.0 if total else 0.0
            ),
            "final_answer_wrong_records": final_answer_wrong_records,
            "final_answer_wrong_records_ratio": (
                final_answer_wrong_records / total * 100.0 if total else 0.0
            ),
            "prove_verify_ratio_distribution_by_record": prove_hist,
            "form_verify_ratio_distribution_by_record": form_hist,
            "global_nodes_total": global_nodes_total,
            "global_prove_required_nodes_total": prove_required_total,
            "global_form_required_nodes_total": form_required_total,
            "global_prove_verified_nodes": prove_verified,
            "global_prove_verified_nodes_ratio": (
                prove_verified / prove_required_total * 100.0 if prove_required_total else 0.0
            ),
            "global_form_verified_nodes": form_verified,
            "global_form_verified_nodes_ratio": (
                form_verified / form_required_total * 100.0 if form_required_total else 0.0
            ),
            "prove_verify_ratio_distribution_top_ids": top_ids,
        }

    return {
        "graph_mode": "fdg",
        "total_records_in_jsonl": len(records),
        "valid_records_with_nodes": sum(
            1 for rec in records if (((rec.get("results") or {}).get("facts") or []))
        ),
        "include_skipped_derived": one(True),
        "exclude_skipped_derived": one(False),
    }


def score_facts(facts: List[JsonDict]) -> JsonDict:
    prove_facts_all = [fact for fact in facts if is_required_fact(fact, include_skipped_derived=True)]
    prove_facts_excluded = [fact for fact in facts if is_required_fact(fact, include_skipped_derived=False)]
    prove_success_nodes = 0
    lean_verify_nodes = 0
    lean_pass_nodes = 0
    for fact in prove_facts_all:
        solved = fact.get("solved_lemma") or {}
        if fact.get("prove_status") == "success":
            prove_success_nodes += 1
        if solved.get("lean_verify") is True:
            lean_verify_nodes += 1
        if solved.get("lean_pass") is True:
            lean_pass_nodes += 1
    return {
        "prove_required_nodes": len(prove_facts_excluded),
        "skip_derived_nodes": len(prove_facts_all) - len(prove_facts_excluded),
        "prove_success_nodes": prove_success_nodes,
        "lean_verify_nodes": lean_verify_nodes,
        "lean_pass_nodes": lean_pass_nodes,
        "success_ratio": prove_success_nodes / len(prove_facts_all) if prove_facts_all else 0.0,
    }


def score_sort_key(row: JsonDict) -> Tuple[float, int, int, int]:
    return (
        -float(row["success_ratio"]),
        -int(row["prove_success_nodes"]),
        -int(row["lean_pass_nodes"]),
        int(row["rollout_id"]),
    )


def summarize_selection(records: List[JsonDict], math_rows: Dict[str, JsonDict]) -> Tuple[JsonDict, List[JsonDict], List[str]]:
    rows: List[JsonDict] = []
    warnings: List[str] = []
    for record in records:
        rid = record_id_of(record)
        parent_id, rollout_id = split_rollout_id(rid)
        facts = ((record.get("results") or {}).get("facts") or [])
        math_row = math_rows.get(rid) or {}
        score = score_facts([fact for fact in facts if isinstance(fact, dict)])
        rows.append(
            {
                "id": rid,
                "parent_id": parent_id,
                "rollout_id": rollout_id,
                "source": math_row.get("source", ""),
                "answer": math_row.get("answer", ""),
                "gold": math_row.get("gold", ""),
                "is_correct": str(math_row.get("is_correct", "0") or "0"),
                **score,
            }
        )
        if not math_row:
            warnings.append(f"missing math_verify row for selected rollout {rid}")

    selected: List[JsonDict] = []
    by_parent: Dict[str, List[JsonDict]] = {}
    for row in rows:
        by_parent.setdefault(str(row["parent_id"]), []).append(row)
    for parent_id in sorted(by_parent):
        best = sorted(by_parent[parent_id], key=score_sort_key)[0]
        selected.append(
            {
                "id": parent_id,
                "parent_id": parent_id,
                "source": best.get("source", ""),
                "selected_rollout_id": int(best["rollout_id"]),
                "selected_record_id": best["id"],
                "is_correct": str(best.get("is_correct", "0") or "0"),
                "score": {
                    key: best[key]
                    for key in (
                        "prove_required_nodes",
                        "prove_success_nodes",
                        "lean_verify_nodes",
                        "lean_pass_nodes",
                        "success_ratio",
                    )
                },
            }
        )

    def accuracy(items: List[JsonDict]) -> float:
        return sum(1 for item in items if str(item.get("is_correct")) == "1") / len(items) if items else math.nan

    per_source: Dict[str, List[JsonDict]] = {}
    for item in selected:
        per_source.setdefault(str(item.get("source") or ""), []).append(item)
    metrics = {
        "total": len(selected),
        "step_proof_best": {
            "accuracy": accuracy(selected),
            "per_source": {source: accuracy(group) for source, group in sorted(per_source.items())},
        },
        "warnings": warnings,
    }
    return metrics, selected, warnings


def failed_nodes_for_classification(records: List[JsonDict]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for record in records:
        rid = record_id_of(record)
        parent_id, rollout_id = split_rollout_id(rid)
        facts = ((record.get("results") or {}).get("facts") or [])
        fact_by_id = {str(f.get("fact_id")): f for f in facts if isinstance(f, dict)}
        for fact in fact_by_id.values():
            if fact_origin(fact) not in {"derived", "answer"}:
                continue
            if int(fact.get("skip", 0) or 0) != 0:
                continue
            if str(fact.get("form_status") or "").strip() != "success":
                continue
            formal = fact.get("formalization") or {}
            if not bool(formal.get("lean_pass")):
                continue
            solved = fact.get("solved_lemma") or {}
            if str(fact.get("prove_status") or "").strip() == "success" and bool(solved.get("lean_verify")):
                continue
            parent_facts = [
                compact_fact(fact_by_id[parent_id])
                for parent_id in fact.get("parent_fact_ids") or []
                if parent_id in fact_by_id
            ]
            rows.append(
                {
                    "record_id": rid,
                    "parent_id": parent_id,
                    "rollout_id": rollout_id,
                    "fact_id": str(fact.get("fact_id") or ""),
                    "node": compact_fact(fact, include_formalization=True, include_proof=True),
                    "parents": parent_facts,
                    "graph_context": {
                        "validation_checks": (record.get("graph") or {}).get("validation_checks")
                        or (record.get("meta") or {}).get("validation_checks")
                        or {},
                        "validation_warnings": (record.get("graph") or {}).get("validation_warnings") or [],
                        "final_fact_ids": (record.get("graph") or {}).get("final_fact_ids") or [],
                        "topo_order": (record.get("graph") or {}).get("topo_order") or [],
                    },
                    "problem": (record.get("input") or {}).get("problem", ""),
                }
            )
    return rows


def truncate_text(value: Any, max_chars: int = 2400) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


def compact_messages(messages: Any, max_chars: int = 1200) -> List[JsonDict]:
    if not isinstance(messages, list):
        return []
    return [
        {"role": msg.get("role", ""), "content": truncate_text(msg.get("content", ""), max_chars)}
        for msg in messages[-4:]
        if isinstance(msg, dict)
    ]


def compact_fact(fact: JsonDict, include_formalization: bool = False, include_proof: bool = False) -> JsonDict:
    payload: JsonDict = {
        "fact_id": fact.get("fact_id", ""),
        "text": truncate_text(fact.get("text", ""), 1200),
        "origin": fact.get("origin", ""),
        "parent_fact_ids": fact.get("parent_fact_ids") or [],
        "is_final_answer": bool(fact.get("is_final_answer")),
        "skip": fact.get("skip", 0),
        "proof_obligation": fact.get("proof_obligation") or {},
        "form_status": fact.get("form_status", ""),
        "prove_status": fact.get("prove_status", ""),
    }
    formal = fact.get("formalization") or {}
    solved = fact.get("solved_lemma") or {}
    if include_formalization:
        payload["formalization"] = {
            "lean_code": truncate_text(formal.get("lean_code", ""), 4000),
            "lean_pass": bool(formal.get("lean_pass")),
            "error_msg": truncate_text(formal.get("error_msg", ""), 1200),
            "tries": formal.get("tries", 0),
            "skipped": bool(formal.get("skipped")),
        }
    else:
        payload["formalization_summary"] = {
            "lean_pass": bool(formal.get("lean_pass")),
            "tries": formal.get("tries", 0),
            "skipped": bool(formal.get("skipped")),
        }
    if include_proof:
        payload["solved_lemma"] = {
            "lean_code": truncate_text(solved.get("lean_code", ""), 4000),
            "lean_pass": bool(solved.get("lean_pass")),
            "lean_verify": bool(solved.get("lean_verify")),
            "error_msg": truncate_text(solved.get("error_msg", ""), 2400),
            "tries": solved.get("tries", 0),
            "skipped": bool(solved.get("skipped")),
            "conversation_tail": compact_messages(
                solved.get("conversation_raw") or solved.get("conversation") or [],
                max_chars=1200,
            ),
        }
    return payload


class TextGenerator:
    def generate_one(self, messages: List[JsonDict]) -> JsonDict:
        raise NotImplementedError


class DryRunGenerator(TextGenerator):
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def generate_one(self, messages: List[JsonDict]) -> JsonDict:
        if self.kind == "classification":
            content = json.dumps(
                {
                    "main_issue": "prover",
                    "secondary_issues": [],
                    "reason_zh": "dry-run placeholder",
                    "evidence": "dry-run placeholder",
                    "confidence": 0.0,
                },
                ensure_ascii=False,
            )
        else:
            content = "```lean4\nby\n  sorry\n```"
        return {
            "ok": True,
            "text": content,
            "message": {"role": "assistant", "content": content},
            "raw_response": {"dry_run": True},
            "prompt_token_overflow": False,
            "output_truncated": False,
        }


class APIGenerator(TextGenerator):
    def __init__(self, cfg: JsonDict, section: str) -> None:
        sec = cfg.get(section) or {}
        api_key = str(sec.get("api_key") or "").strip() or os.getenv(str(sec.get("api_key_env") or ""), "")
        if not api_key:
            raise RuntimeError(f"{section} API key is missing; set {section}.api_key or {section}.api_key_env")
        self.model = str(sec.get("model"))
        self.base_url = str(sec.get("api_base_url")).rstrip("/")
        self.api_key = api_key
        self.timeout = float(sec.get("api_timeout", 300))
        self.max_retries = int(sec.get("api_max_retries", 3))
        self.retry_sleep = float(sec.get("api_retry_sleep", 2.0))
        self.max_tokens = int(sec.get("max_tokens", 4096))
        self.temperature = float(sec.get("temperature", 0.0))
        self.top_p = float(sec.get("top_p", 1.0))

    def _request(self, messages: List[JsonDict], token_param: str) -> JsonDict:
        payload = {
            "model": self.model,
            "messages": messages,
            token_param: self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate_one(self, messages: List[JsonDict]) -> JsonDict:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                try:
                    raw = self._request(messages, "max_tokens")
                    token_param = "max_tokens"
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    if "max_completion_tokens" not in body or "max_tokens" not in body:
                        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
                    raw = self._request(messages, "max_completion_tokens")
                    token_param = "max_completion_tokens"
                choice = (raw.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "")
                return {
                    "ok": True,
                    "text": content,
                    "message": {"role": "assistant", "content": content},
                    "raw_response": raw,
                    "api_token_param": token_param,
                    "prompt_token_overflow": False,
                    "output_truncated": "length" in finish_reason.lower(),
                }
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
        return {
            "ok": False,
            "text": "",
            "message": {"role": "assistant", "content": ""},
            "raw_response": {"error": last_error},
            "error": last_error,
            "prompt_token_overflow": False,
            "output_truncated": False,
        }


class VLLMGenerator(TextGenerator):
    def __init__(self, cfg: JsonDict, section: str) -> None:
        sec = cfg.get(section) or {}
        from proofflow.local_vllm import LocalLLMManager

        os.environ["CUDA_VISIBLE_DEVICES"] = str(sec.get("gpus", ""))
        self.manager = LocalLLMManager(
            model_path=str(sec.get("vllm_model_path")),
            tensor_parallel_size=int(sec.get("tensor_parallel_size", 1)),
            max_tokens=int(sec.get("max_tokens", 4096)),
            temperature=float(sec.get("temperature", 0.0)),
            top_p=float(sec.get("top_p", 1.0)),
            presence_penalty=float(sec.get("presence_penalty", 0.0)),
            frequency_penalty=float(sec.get("frequency_penalty", 0.0)),
            seed=int((cfg.get("run") or {}).get("seed", 0)),
            top_k=int(sec.get("top_k", 20)),
            token_limit=int(sec.get("token_limit", 8192)),
            dtype=str(sec.get("dtype", "float16")),
            gpu_memory_utilization=float(sec.get("gpu_memory_utilization", 0.95)),
            chat_template_kwargs=sec.get("chat_template_kwargs") or {},
        )

    def generate_one(self, messages: List[JsonDict]) -> JsonDict:
        result = self.manager.batch_generate_with_metadata([messages])[0]
        text = result.get("text") or ""
        return {
            "ok": not bool(result.get("prompt_token_overflow")),
            "text": text,
            "message": {"role": "assistant", "content": text},
            "raw_response": result,
            **result,
        }


def make_generator(cfg: JsonDict, section: str, kind: str) -> TextGenerator:
    sec = cfg.get(section) or {}
    if bool(sec.get("dry_run", False)):
        return DryRunGenerator(kind)
    backend = str(sec.get("backend", "api")).strip().lower()
    if backend == "api":
        return APIGenerator(cfg, section)
    if backend == "vllm":
        return VLLMGenerator(cfg, section)
    raise ValueError(f"unsupported {section}.backend: {backend}")


def build_classification_messages(item: JsonDict) -> List[JsonDict]:
    payload = {
        "record_id": item["record_id"],
        "parent_id": item["parent_id"],
        "rollout_id": item["rollout_id"],
        "problem": truncate_text(item.get("problem", ""), 2400),
        "graph_context": item.get("graph_context") or {},
        "node": item["node"],
        "parents": item.get("parents") or [],
    }
    system = (
        "You are a Lean 4, Mathlib, and step-proof debugging assistant. "
        "Classify the root cause stage for one DAG node whose formalization type-checked "
        "but prover verification failed. The formalization theorem body may contain by sorry; "
        "that placeholder is expected and is not itself a failure cause."
    )
    user = (
        "Classify this failed node into exactly one main_issue: builder, formalizer, or prover.\n"
        "builder means the DAG/dependencies/parent facts are missing or wrong.\n"
        "formalizer means the Lean statement type-checks but is semantically wrong, too strong, "
        "underspecified, or not faithful to the informal node.\n"
        "prover means the statement appears reasonable and the failure is mainly proof-generation ability, "
        "invalid tactics, truncation, looping, or inability to find Mathlib lemmas.\n"
        "secondary_issues may contain zero or more of builder/formalizer/prover.\n"
        "Return only JSON with fields: main_issue, secondary_issues, reason_zh, evidence, confidence.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> JsonDict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_classification(parsed: JsonDict) -> JsonDict:
    valid = {"builder", "formalizer", "prover"}
    main = str(parsed.get("main_issue") or "").strip().lower()
    if main not in valid:
        main = "prover"
    secondary = parsed.get("secondary_issues") or []
    if not isinstance(secondary, list):
        secondary = [secondary]
    secondary = [str(item).strip().lower() for item in secondary if str(item).strip().lower() in valid]
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "main_issue": main,
        "secondary_issues": secondary,
        "reason_zh": str(parsed.get("reason_zh") or ""),
        "evidence": str(parsed.get("evidence") or ""),
        "confidence": confidence,
    }


def classify_nodes(cfg: JsonDict, failed_items: List[JsonDict], out_path: Path) -> List[JsonDict]:
    force = bool((cfg.get("run") or {}).get("force", False))
    resume = bool((cfg.get("run") or {}).get("resume", True))
    existing: Dict[str, JsonDict] = {}
    if resume and not force and out_path.is_file():
        for row in read_jsonl(out_path):
            existing[node_key(str(row.get("record_id")), str(row.get("fact_id")))] = row

    generator = make_generator(cfg, "classification", "classification")
    rows: List[JsonDict] = []
    for idx, item in enumerate(failed_items, 1):
        key = node_key(item["record_id"], item["fact_id"])
        if key in existing:
            rows.append(existing[key])
            continue
        messages = build_classification_messages(item)
        print(f"[classify {idx}/{len(failed_items)}] {key}")
        result = generator.generate_one(messages)
        try:
            parsed = normalize_classification(extract_json_object(result.get("text") or ""))
            parse_error = ""
        except Exception as exc:
            parsed = normalize_classification(
                {
                    "main_issue": "prover",
                    "secondary_issues": [],
                    "reason_zh": "分类输出无法解析，默认作为 prover 主因进入 retry。",
                    "evidence": str(exc),
                    "confidence": 0.0,
                }
            )
            parse_error = str(exc)
        row = {
            "record_id": item["record_id"],
            "parent_id": item["parent_id"],
            "rollout_id": item["rollout_id"],
            "fact_id": item["fact_id"],
            **parsed,
            "request_payload": item,
            "request_messages": messages,
            "response_message": result.get("message") or {"role": "assistant", "content": result.get("text", "")},
            "conversation_messages": messages
            + [result.get("message") or {"role": "assistant", "content": result.get("text", "")}],
            "raw_response": result.get("raw_response"),
            "parse_error": parse_error,
        }
        rows.append(row)
    write_jsonl(out_path, rows)
    return rows


def split_lean_header_body(lean_code: str) -> JsonDict:
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
    return {"lean_header": "\n".join(header_lines).strip(), "lean_body": "\n".join(body_lines).strip()}


def load_prompt_messages(prompt_name: str, lean_code: str) -> List[JsonDict]:
    parts = split_lean_header_body(lean_code)
    system_path = PROMPT_ROOT / "system" / f"{prompt_name}.md"
    user_path = PROMPT_ROOT / "user" / f"{prompt_name}.md"
    if not user_path.is_file():
        raise FileNotFoundError(f"missing prover user prompt: {user_path}")
    user = user_path.read_text(encoding="utf-8").format(**parts)
    messages: List[JsonDict] = []
    if system_path.is_file():
        system = system_path.read_text(encoding="utf-8").strip()
        if system:
            messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def theorem_prefix(lean_code: str) -> str:
    match = re.search(r":=\s*by\b", lean_code)
    if not match:
        raise ValueError("formalization.lean_code does not contain ':= by'")
    return lean_code[: match.start()].rstrip()


def extract_lean_block(text: str) -> str:
    matches = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_proof_body(text: str) -> str:
    candidate = extract_lean_block(text)
    match = re.search(r":=\s*by\b", candidate)
    if match:
        body = candidate[match.end() :].strip()
    else:
        body = candidate.strip()
    body = re.sub(r"^by\b", "", body).strip()
    body = re.sub(r"^```(?:lean4|lean)?\s*", "", body, flags=re.IGNORECASE).strip()
    body = re.sub(r"\s*```$", "", body).strip()
    return body or "sorry"


def splice_proof(formal_lean_code: str, assistant_text: str) -> Tuple[str, str]:
    prefix = theorem_prefix(formal_lean_code)
    proof_body = extract_proof_body(assistant_text)
    return f"{prefix} := by\n{proof_body}", proof_body


def retry_user_message(error_msg: Any, max_error_chars: int) -> JsonDict:
    error_text = truncate_text(error_msg, max_error_chars)
    return {
        "role": "user",
        "content": (
            "The previous Lean4 proof did not verify. Below is the kimina-lean-server error output.\n"
            "Keep the original theorem statement unchanged. Correct only the proof.\n\n"
            f"Lean error/warnings:\n{error_text}\n\n"
            "Based on the full conversation above and these errors, please correct the previous response."
        ),
    }


def make_lean_server(cfg: JsonDict):
    from proofflow.lean_check import LeanServer

    lean_cfg = cfg.get("lean_runtime") or {}
    return LeanServer(
        api_url=str(lean_cfg.get("lean_api_url") or "http://localhost:8000"),
        backend="kimina_server",
        api_key_env=str(lean_cfg.get("lean_api_key_env") or "KIMINA_API_KEY"),
        server_timeout=int(lean_cfg.get("lean_server_timeout", 300)),
        server_reuse=bool(lean_cfg.get("lean_server_reuse", True)),
        server_debug=bool(lean_cfg.get("lean_server_debug", False)),
    )


def retry_prover_nodes(
    cfg: JsonDict,
    records: List[JsonDict],
    classifications: List[JsonDict],
    out_path: Path,
) -> List[JsonDict]:
    force = bool((cfg.get("run") or {}).get("force", False))
    resume = bool((cfg.get("run") or {}).get("resume", True))
    existing: Dict[str, JsonDict] = {}
    if resume and not force and out_path.is_file():
        for row in read_jsonl(out_path):
            existing[node_key(str(row.get("record_id")), str(row.get("fact_id")))] = row

    retry_targets = [row for row in classifications if row.get("main_issue") == "prover"]
    record_by_id = {record_id_of(record): record for record in records}
    generator = make_generator(cfg, "retry", "retry")
    dry_run = bool((cfg.get("retry") or {}).get("dry_run", False))
    lean_server = None if dry_run else make_lean_server(cfg)
    prompt_name = str((cfg.get("retry") or {}).get("prompt", "prove.paper_goedel_v2"))
    max_retries = int((cfg.get("retry") or {}).get("max_retries", 3))
    max_error_chars = int((cfg.get("retry") or {}).get("max_error_chars", 1200))
    rows: List[JsonDict] = []

    for idx, cls in enumerate(retry_targets, 1):
        key = node_key(str(cls["record_id"]), str(cls["fact_id"]))
        if key in existing:
            rows.append(existing[key])
            continue
        print(f"[retry {idx}/{len(retry_targets)}] {key}")
        record = record_by_id[str(cls["record_id"])]
        facts = ((record.get("results") or {}).get("facts") or [])
        fact = next((f for f in facts if isinstance(f, dict) and str(f.get("fact_id")) == str(cls["fact_id"])), None)
        if fact is None:
            rows.append(
                {
                    "record_id": cls["record_id"],
                    "parent_id": cls["parent_id"],
                    "rollout_id": cls["rollout_id"],
                    "fact_id": cls["fact_id"],
                    "success": False,
                    "success_bucket": "failed",
                    "attempts": [],
                    "error": "fact not found",
                }
            )
            continue
        formal_lean_code = str(((fact.get("formalization") or {}).get("lean_code") or ""))
        messages = load_prompt_messages(prompt_name, formal_lean_code)
        attempts: List[JsonDict] = []
        success = False
        success_bucket = "failed"
        final_lean_result: JsonDict = {"lean_pass": False, "lean_verify": False, "error_msg": ""}

        for attempt_idx in range(max_retries + 1):
            result = generator.generate_one(messages)
            assistant = result.get("message") or {"role": "assistant", "content": result.get("text", "")}
            try:
                candidate_lean_code, proof_body = splice_proof(formal_lean_code, assistant.get("content", ""))
                splice_error = ""
            except Exception as exc:
                candidate_lean_code = formal_lean_code
                proof_body = ""
                splice_error = str(exc)

            if dry_run:
                lean_pass, lean_verify, error_msg = False, False, "dry-run: skipped kimina verification"
            elif splice_error:
                lean_pass, lean_verify, error_msg = False, False, splice_error
            else:
                assert lean_server is not None
                lean_pass, lean_verify, error_msg = lean_server.check_lean_string(candidate_lean_code)

            final_lean_result = {
                "lean_pass": bool(lean_pass),
                "lean_verify": bool(lean_verify),
                "error_msg": error_msg,
            }
            attempt_row = {
                "attempt_index": attempt_idx,
                "retry_count": attempt_idx,
                "request_messages": copy.deepcopy(messages),
                "response_message": assistant,
                "conversation_messages": copy.deepcopy(messages) + [assistant],
                "raw_response": result.get("raw_response"),
                "candidate_lean_code": candidate_lean_code,
                "proof_body": proof_body,
                "splice_error": splice_error,
                "lean_result": final_lean_result,
                "output_truncated": bool(result.get("output_truncated")),
                "prompt_token_overflow": bool(result.get("prompt_token_overflow")),
            }
            attempts.append(attempt_row)
            messages = copy.deepcopy(messages) + [assistant]
            if lean_verify:
                success = True
                success_bucket = str(attempt_idx)
                break
            if attempt_idx < max_retries:
                messages.append(retry_user_message(error_msg, max_error_chars))

        rows.append(
            {
                "record_id": cls["record_id"],
                "parent_id": cls["parent_id"],
                "rollout_id": cls["rollout_id"],
                "fact_id": cls["fact_id"],
                "classification": {
                    "main_issue": cls.get("main_issue"),
                    "secondary_issues": cls.get("secondary_issues") or [],
                    "reason_zh": cls.get("reason_zh", ""),
                    "evidence": cls.get("evidence", ""),
                    "confidence": cls.get("confidence", 0.0),
                },
                "success": success,
                "success_bucket": success_bucket,
                "attempt_count": len(attempts),
                "final_lean_result": final_lean_result,
                "attempts": attempts,
            }
        )

    write_jsonl(out_path, rows)
    return rows


def apply_retry_results(records: List[JsonDict], retry_rows: List[JsonDict]) -> List[JsonDict]:
    updated = copy.deepcopy(records)
    record_by_id = {record_id_of(record): record for record in updated}
    retry_by_key = {node_key(str(row.get("record_id")), str(row.get("fact_id"))): row for row in retry_rows}
    for key, row in retry_by_key.items():
        record_id, fact_id = key.split("::", 1)
        record = record_by_id.get(record_id)
        if record is None:
            continue
        facts = ((record.get("results") or {}).get("facts") or [])
        fact = next((f for f in facts if isinstance(f, dict) and str(f.get("fact_id")) == fact_id), None)
        if fact is None:
            continue
        attempts = row.get("attempts") or []
        fact["prove_failure_retry"] = {
            "main_issue": ((row.get("classification") or {}).get("main_issue")),
            "success": bool(row.get("success")),
            "success_bucket": row.get("success_bucket"),
            "attempt_count": row.get("attempt_count", len(attempts)),
            "final_lean_result": row.get("final_lean_result") or {},
        }
        if bool(row.get("success")) and attempts:
            successful_attempt = next(
                (attempt for attempt in attempts if bool(((attempt.get("lean_result") or {}).get("lean_verify")))),
                attempts[-1],
            )
            fact["prove_status"] = "success"
            fact["solved_lemma"] = {
                "lean_code": successful_attempt.get("candidate_lean_code", ""),
                "lean_pass": True,
                "lean_verify": True,
                "error_msg": (successful_attempt.get("lean_result") or {}).get("error_msg"),
                "tries": int(row.get("attempt_count", len(attempts))),
                "skipped": False,
                "conversation": successful_attempt.get("conversation_messages") or [],
                "conversation_raw": successful_attempt.get("conversation_messages") or [],
            }
    return updated


def retry_scaling_stats(retry_rows: List[JsonDict]) -> JsonDict:
    buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
    for row in retry_rows:
        key = str(row.get("success_bucket") or "failed")
        if key not in buckets:
            key = "failed"
        buckets[key] += 1
    total = sum(buckets.values())
    return {
        "total_prover_main_issue_nodes": total,
        "buckets": {
            key: {"count": count, "ratio": (count / total if total else 0.0)}
            for key, count in buckets.items()
        },
    }


def before_after_summary(before_stats: JsonDict, after_stats: JsonDict, before_metrics: JsonDict, after_metrics: JsonDict) -> JsonDict:
    metrics: JsonDict = {}
    for scope in ("include_skipped_derived", "exclude_skipped_derived"):
        for name in ("global_prove_verified_nodes_ratio", "global_form_verified_nodes_ratio"):
            key = f"{scope}.{name}"
            before = float(((before_stats.get(scope) or {}).get(name) or 0.0))
            after = float(((after_stats.get(scope) or {}).get(name) or 0.0))
            metrics[key] = {"before": before, "after": after, "delta": after - before}
    before_acc = float(((before_metrics.get("step_proof_best") or {}).get("accuracy") or 0.0))
    after_acc = float(((after_metrics.get("step_proof_best") or {}).get("accuracy") or 0.0))
    metrics["step_proof_best.accuracy"] = {
        "before": before_acc,
        "after": after_acc,
        "delta": after_acc - before_acc,
    }
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    out_root = output_root(cfg)
    selected_dir = out_root / "selected"
    analysis_dir = out_root / "analysis"
    stage3_dir = out_root / "step_proof_results" / "result_stage3"
    stats_dir = out_root / "step_proof_results" / "stats"
    summary_dir = out_root / "summary"

    records, math_rows = load_source_records(cfg)
    selected_records, selection_meta = select_subset(cfg, records, math_rows)
    top_n = int((cfg.get("stats") or {}).get("top_n_per_bucket", 10))
    selected_math_rows = [
        math_rows[record_id_of(record)]
        for record in selected_records
        if record_id_of(record) in math_rows
    ]
    write_jsonl(selected_dir / "stage3_before.jsonl", selected_records)
    write_jsonl(selected_dir / "all_rollouts_eval.jsonl", selected_math_rows)
    write_json(summary_dir / "selection_meta.json", selection_meta)
    print(
        "[prepare] "
        f"selected_problems={selection_meta['selected_problems']} "
        f"selected_records={selection_meta['selected_records']} "
        f"rejected_unclosed={selection_meta['rejected_unclosed_think_problem_count']}"
    )

    before_stats = compute_stage3_stats(selected_records, top_n=top_n)
    before_metrics, _before_selected, before_warnings = summarize_selection(selected_records, math_rows)
    write_json(stats_dir / "stage3_verify_stats_before.json", before_stats)
    write_json(summary_dir / "metrics_before.json", before_metrics)

    failed_items = failed_nodes_for_classification(selected_records)
    print(f"[classify] eligible failed nodes={len(failed_items)}")
    classifications = classify_nodes(cfg, failed_items, analysis_dir / "node_issue_classifications.jsonl")

    retry_rows = retry_prover_nodes(
        cfg,
        selected_records,
        classifications,
        analysis_dir / "prover_retry_attempts.jsonl",
    )
    after_records = apply_retry_results(selected_records, retry_rows)
    write_jsonl(stage3_dir / "stage3_results.jsonl", after_records)

    after_stats = compute_stage3_stats(after_records, top_n=top_n)
    after_metrics, _after_selected, after_warnings = summarize_selection(after_records, math_rows)
    write_json(stats_dir / "stage3_verify_stats.json", after_stats)
    write_json(summary_dir / "metrics.json", after_metrics)
    write_json(summary_dir / "retry_scaling_stats.json", retry_scaling_stats(retry_rows))
    compare = before_after_summary(before_stats, after_stats, before_metrics, after_metrics)
    compare["warnings"] = before_warnings + after_warnings
    write_json(summary_dir / "before_after_metrics.json", compare)
    print(f"[done] output -> {out_root}")
    print(json.dumps(compare, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
