#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proofflow.graph_mode import FDG_GRAPH_MODE, ensure_single_graph_mode, extract_record_items
from proofflow.vis import (
    build_dag,
    create_interactive_graph_only_visualization,
    create_interactive_visualization,
)


JsonDict = Dict[str, Any]
COMPARE_FIELD_OPTIONS = [
    ("text", "Fact text"),
    ("origin", "Origin"),
    ("proof_obligation.informal_statement_content", "Proof obligation"),
    ("formalization.lean_code", "Formalization Lean"),
    ("formalization.dependency_context_block", "Context block"),
    ("formalization.error_msg", "Formalization error"),
    ("solved_lemma.lean_code", "Solved lemma Lean"),
    ("solved_lemma.error_msg", "Prove error"),
    ("formalization.raw_conv", "Formalization raw conv"),
    ("solved_lemma.conversation_raw", "Prove conversation raw"),
]
REQUIRED_FILES = [
    "scores.jsonl",
    "selected_step_proof.jsonl",
    "math_verify/all_rollouts_eval.jsonl",
    "math_verify/step_proof_best_eval.jsonl",
    "summary/metrics.json",
    "step_proof_results/result_stage3/stage3_results.jsonl",
]


def _load_jsonl(path: Path) -> List[JsonDict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_bool_correct(value: Any) -> bool:
    return str(value).strip() == "1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _score_sort_key(row: JsonDict) -> tuple:
    return (
        -_safe_float(row.get("success_ratio")),
        -_safe_int(row.get("prove_success_nodes")),
        -_safe_int(row.get("lean_pass_nodes")),
        _safe_int(row.get("rollout_id"), 10**9),
    )


def _score_rank_key(row: JsonDict) -> tuple:
    return (
        _safe_float(row.get("success_ratio")),
        _safe_int(row.get("prove_success_nodes")),
        _safe_int(row.get("lean_pass_nodes")),
        -_safe_int(row.get("rollout_id"), 10**9),
    )


def _parent_id(row: JsonDict) -> str:
    parent = str(row.get("parent_id") or "").strip()
    if parent:
        return parent
    rid = str(row.get("id") or "").strip()
    if "__rollout_" in rid:
        return rid.rsplit("__rollout_", 1)[0]
    return rid


def _rollout_id(row: JsonDict) -> Optional[int]:
    if str(row.get("rollout_id", "")).strip():
        try:
            return int(row["rollout_id"])
        except (TypeError, ValueError):
            pass
    rid = str(row.get("id") or "")
    if "__rollout_" in rid:
        try:
            return int(rid.rsplit("__rollout_", 1)[-1])
        except ValueError:
            return None
    return None


def _fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _get_nested_value(payload: JsonDict, field_path: str) -> Any:
    if field_path == "formalization.raw_conv":
        formalization = payload.get("formalization") if isinstance(payload, dict) else {}
        if not isinstance(formalization, dict):
            return ""
        for key in ("raw_conv", "conversation_raw", "conversation"):
            value = formalization.get(key)
            if value not in (None, ""):
                return value
        return ""
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
        if current is None:
            return ""
    return current


def _extract_nodes(rec: JsonDict, source: str) -> List[JsonDict]:
    nodes, graph_mode = extract_record_items(rec, source)
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"selected record has empty {source} items")
    if graph_mode != FDG_GRAPH_MODE:
        raise ValueError(f"only FDG records are supported, got graph_mode={graph_mode!r}")
    return nodes


def _build_proof_str(rec: JsonDict) -> str:
    inp = rec.get("input", {})
    problem = str(inp.get("problem", "")).strip()
    raw_cot = str(inp.get("raw_cot", "")).strip()
    answer = str(inp.get("answer", "")).strip()
    parts: List[str] = []
    if problem:
        parts.extend(["Problem:", problem])
    if raw_cot:
        parts.extend(["", "Raw CoT:", raw_cot])
    if answer and not raw_cot:
        parts.extend(["", "Answer:", answer])
    return "\n".join(parts)


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _format_conversation(value: Any) -> str:
    if not isinstance(value, list):
        return _text_value(value)
    chunks: List[str] = []
    for idx, message in enumerate(value, start=1):
        if isinstance(message, dict):
            role = str(message.get("role") or f"message_{idx}")
            content = _text_value(message.get("content"))
        else:
            role = f"message_{idx}"
            content = _text_value(message)
        chunks.append(f"### {role}\n{content}".rstrip())
    return "\n\n".join(chunks)


def _stage3_items(rec: JsonDict, source: str) -> List[JsonDict]:
    try:
        return _extract_nodes(rec, source)
    except Exception:
        pass
    for key in ("results", "graph"):
        items = rec.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _record_id_for(parent_id: str, rollout_id: int) -> str:
    return f"{parent_id}__rollout_{rollout_id}"


def _stage1_signature(rec: JsonDict) -> List[JsonDict]:
    nodes = _extract_nodes(rec, "graph")
    signature: List[JsonDict] = []
    for node in nodes:
        node_id = str(node.get("fact_id") or node.get("id") or "").strip()
        if not node_id:
            continue
        signature.append(
            {
                "fact_id": node_id,
                "text": str(node.get("text") or "").strip(),
                "parent_fact_ids": sorted(
                    str(parent).strip()
                    for parent in node.get("parent_fact_ids") or []
                    if str(parent).strip()
                ),
                "origin": str(node.get("origin") or "").strip(),
                "is_final_answer": bool(node.get("is_final_answer")),
            }
        )
    return sorted(signature, key=lambda item: _node_sort_key(str(item.get("fact_id") or "")))


def _node_sort_key(node_id: str) -> tuple:
    prefix = "".join(ch for ch in node_id if not ch.isdigit())
    digits = "".join(ch for ch in node_id if ch.isdigit())
    return (prefix, int(digits) if digits else -1, node_id)


def _field_label(field_path: str) -> str:
    for value, label in COMPARE_FIELD_OPTIONS:
        if value == field_path:
            return label
    return field_path


@dataclass
class ExperimentStatus:
    name: str
    path: Path
    complete: bool
    missing: List[str]
    metrics: JsonDict

    def to_payload(self) -> JsonDict:
        return {
            "name": self.name,
            "path": str(self.path),
            "complete": self.complete,
            "missing": self.missing,
            "metrics": self.metrics,
            "summary": _metrics_summary(self.metrics),
        }


@dataclass
class ExperimentData:
    status: ExperimentStatus
    scores_by_parent: Dict[str, List[JsonDict]]
    scores_by_id: Dict[str, JsonDict]
    eval_by_rollout: Dict[str, JsonDict]
    selected_eval_by_parent: Dict[str, JsonDict]
    pass_at_1_by_parent: Dict[str, bool]
    pass_at_k_by_parent: Dict[str, bool]
    parent_source: Dict[str, str]


def _metrics_summary(metrics: JsonDict) -> JsonDict:
    pass_at_4 = metrics.get("pass_at_4") or metrics.get("pass_at_k") or {}
    selection = metrics.get("selection") or {}
    hit_rate = ""
    for key, value in selection.items():
        if key.startswith("hit_rate_on_"):
            hit_rate = value
            break
    return {
        "total": metrics.get("total", ""),
        "pass_at_1": _fmt_pct((metrics.get("pass_at_1") or {}).get("accuracy")),
        "pass_at_4": _fmt_pct(pass_at_4.get("accuracy")),
        "ours": _fmt_pct((metrics.get("step_proof_best") or {}).get("accuracy")),
        "selection_hit_rate": _fmt_pct(hit_rate),
    }


class StepProofCompareApp:
    def __init__(self, repo_root: Path, results_root: Path, source: str, graph_only: bool) -> None:
        self.repo_root = repo_root
        self.results_root = results_root
        self.source = source
        self.graph_only = graph_only
        self.cache_dir = repo_root / ".tmp_step_proof_compare_html"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()
        self._status_cache: Optional[List[ExperimentStatus]] = None
        self._data_cache: Dict[str, ExperimentData] = {}
        self._stage1_cache: Dict[str, Dict[str, JsonDict]] = {}
        self._stage2_cache: Dict[str, Dict[str, JsonDict]] = {}
        self._stage3_cache: Dict[str, Dict[str, JsonDict]] = {}

    def scan_experiments(self) -> List[ExperimentStatus]:
        if self._status_cache is not None:
            return self._status_cache
        statuses: List[ExperimentStatus] = []
        if not self.results_root.is_dir():
            self._status_cache = []
            return []
        for exp_dir in sorted([p for p in self.results_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            missing = [
                rel
                for rel in REQUIRED_FILES
                if not (exp_dir / rel).is_file() or (exp_dir / rel).stat().st_size <= 0
            ]
            metrics: JsonDict = {}
            metrics_path = exp_dir / "summary" / "metrics.json"
            if metrics_path.is_file() and metrics_path.stat().st_size > 0:
                try:
                    metrics = _read_json(metrics_path)
                except Exception as exc:
                    missing.append(f"summary/metrics.json ({exc})")
            statuses.append(
                ExperimentStatus(
                    name=exp_dir.name,
                    path=exp_dir,
                    complete=not missing,
                    missing=missing,
                    metrics=metrics,
                )
            )
        self._status_cache = statuses
        return statuses

    def complete_experiment_names(self) -> List[str]:
        return [status.name for status in self.scan_experiments() if status.complete]

    def _status_by_name(self, exp_name: str) -> ExperimentStatus:
        for status in self.scan_experiments():
            if status.name == exp_name:
                if not status.complete:
                    raise ValueError(f"experiment is incomplete: {exp_name}")
                return status
        raise KeyError(f"unknown experiment: {exp_name}")

    def _load_data(self, exp_name: str) -> ExperimentData:
        if exp_name in self._data_cache:
            return self._data_cache[exp_name]

        status = self._status_by_name(exp_name)
        exp_dir = status.path
        score_rows = _load_jsonl(exp_dir / "scores.jsonl")
        eval_rows = _load_jsonl(exp_dir / "math_verify" / "all_rollouts_eval.jsonl")
        selected_rows = _load_jsonl(exp_dir / "math_verify" / "step_proof_best_eval.jsonl")

        scores_by_parent: Dict[str, List[JsonDict]] = {}
        scores_by_id: Dict[str, JsonDict] = {}
        parent_source: Dict[str, str] = {}
        for row in score_rows:
            rid = str(row.get("id") or "").strip()
            parent = _parent_id(row)
            if not rid or not parent:
                continue
            row = dict(row)
            row["parent_id"] = parent
            row["rollout_id"] = _safe_int(row.get("rollout_id"))
            scores_by_parent.setdefault(parent, []).append(row)
            scores_by_id[rid] = row
            if row.get("source"):
                parent_source.setdefault(parent, str(row.get("source") or ""))
        for rows in scores_by_parent.values():
            rows.sort(key=_score_sort_key)

        eval_by_rollout: Dict[str, JsonDict] = {}
        pass_at_1_by_parent: Dict[str, bool] = {}
        pass_at_k_by_parent: Dict[str, bool] = {}
        for row in eval_rows:
            rid = str(row.get("id") or "").strip()
            parent = _parent_id(row)
            if not rid or not parent:
                continue
            eval_by_rollout[rid] = row
            ok = _to_bool_correct(row.get("is_correct"))
            pass_at_k_by_parent[parent] = pass_at_k_by_parent.get(parent, False) or ok
            if _rollout_id(row) == 1:
                pass_at_1_by_parent[parent] = ok
            if row.get("source"):
                parent_source.setdefault(parent, str(row.get("source") or ""))

        selected_eval_by_parent: Dict[str, JsonDict] = {}
        for row in selected_rows:
            parent = _parent_id(row)
            if parent:
                selected_eval_by_parent[parent] = row
                if row.get("source"):
                    parent_source.setdefault(parent, str(row.get("source") or ""))

        data = ExperimentData(
            status=status,
            scores_by_parent=scores_by_parent,
            scores_by_id=scores_by_id,
            eval_by_rollout=eval_by_rollout,
            selected_eval_by_parent=selected_eval_by_parent,
            pass_at_1_by_parent=pass_at_1_by_parent,
            pass_at_k_by_parent=pass_at_k_by_parent,
            parent_source=parent_source,
        )
        self._data_cache[exp_name] = data
        return data

    def _load_stage3_records(self, exp_name: str) -> Dict[str, JsonDict]:
        if exp_name in self._stage3_cache:
            return self._stage3_cache[exp_name]
        status = self._status_by_name(exp_name)
        stage3_path = status.path / "step_proof_results" / "result_stage3" / "stage3_results.jsonl"
        rows = _load_jsonl(stage3_path)
        ensure_single_graph_mode(rows, source_name=str(stage3_path))
        records: Dict[str, JsonDict] = {}
        for rec in rows:
            rid = str((rec.get("meta") or {}).get("record_id") or "").strip()
            if rid:
                records[rid] = rec
        self._stage3_cache[exp_name] = records
        return records

    def _load_stage1_records(self, exp_name: str) -> Dict[str, JsonDict]:
        if exp_name in self._stage1_cache:
            return self._stage1_cache[exp_name]
        status = self._status_by_name(exp_name)
        stage1_path = status.path / "step_proof_results" / "result_stage1" / "graphs.jsonl"
        if not stage1_path.is_file() or stage1_path.stat().st_size <= 0:
            self._stage1_cache[exp_name] = {}
            return {}
        rows = _load_jsonl(stage1_path)
        ensure_single_graph_mode(rows, source_name=str(stage1_path))
        records: Dict[str, JsonDict] = {}
        for rec in rows:
            rid = str((rec.get("meta") or {}).get("record_id") or "").strip()
            if rid:
                records[rid] = rec
        self._stage1_cache[exp_name] = records
        return records

    def _load_stage2_records(self, exp_name: str) -> Dict[str, JsonDict]:
        if exp_name in self._stage2_cache:
            return self._stage2_cache[exp_name]
        status = self._status_by_name(exp_name)
        stage2_path = status.path / "step_proof_results" / "result_stage2" / "stage2_results.jsonl"
        if not stage2_path.is_file() or stage2_path.stat().st_size <= 0:
            self._stage2_cache[exp_name] = {}
            return {}
        rows = _load_jsonl(stage2_path)
        ensure_single_graph_mode(rows, source_name=str(stage2_path))
        records: Dict[str, JsonDict] = {}
        for rec in rows:
            rid = str((rec.get("meta") or {}).get("record_id") or "").strip()
            if rid:
                records[rid] = rec
        self._stage2_cache[exp_name] = records
        return records

    def experiments_payload(self) -> JsonDict:
        statuses = self.scan_experiments()
        return {
            "root": str(self.results_root),
            "experiments": [status.to_payload() for status in statuses],
            "complete_names": [status.name for status in statuses if status.complete],
        }

    def _selected_info(self, data: ExperimentData, parent_id: str) -> JsonDict:
        selected = data.selected_eval_by_parent.get(parent_id) or {}
        rollout = _rollout_id(selected)
        rid = ""
        if rollout is not None:
            rid = f"{parent_id}__rollout_{rollout}"
        score = data.scores_by_id.get(rid, {})
        return {
            "rollout_id": rollout if rollout is not None else "",
            "record_id": rid,
            "is_correct": _to_bool_correct(selected.get("is_correct")),
            "success_ratio": score.get("success_ratio", ""),
            "prove_success_nodes": score.get("prove_success_nodes", ""),
            "lean_pass_nodes": score.get("lean_pass_nodes", ""),
        }

    def _case_payload(self, parent_id: str, exp_names: List[str]) -> JsonDict:
        loaded = {name: self._load_data(name) for name in exp_names}
        first = loaded[exp_names[0]]
        first_row = (first.scores_by_parent.get(parent_id) or [{}])[0]
        by_exp: Dict[str, JsonDict] = {}
        correct_rollouts: Dict[str, List[int]] = {}
        for name, data in loaded.items():
            by_exp[name] = self._selected_info(data, parent_id)
            correct_rollouts[name] = sorted(
                [
                    rid
                    for rid, row in data.eval_by_rollout.items()
                    if _parent_id(row) == parent_id and _to_bool_correct(row.get("is_correct"))
                    for rid in [_rollout_id(row)]
                    if rid is not None
                ]
            )
        return {
            "parent_id": parent_id,
            "source": first.parent_source.get(parent_id, ""),
            "question": str(first_row.get("question") or "")[:240],
            "gold": first_row.get("gold", ""),
            "by_exp": by_exp,
            "correct_rollouts": correct_rollouts,
        }

    def mine_cases(self, exp_names: List[str], limit: int = 80) -> JsonDict:
        exp_names = self._normalize_exp_names(exp_names)
        loaded = {name: self._load_data(name) for name in exp_names}
        parent_sets = [set(data.scores_by_parent.keys()) for data in loaded.values()]
        common_parents = sorted(set.intersection(*parent_sets)) if parent_sets else []

        buckets: Dict[str, List[JsonDict]] = {
            "pass4_correct_selected_wrong": [],
            "pass1_correct_selected_wrong": [],
            "a_wrong_b_correct": [],
            "a_correct_b_wrong": [],
            "high_score_wrong": [],
        }

        for parent in common_parents:
            selected_ok = {
                name: self._selected_info(data, parent)["is_correct"]
                for name, data in loaded.items()
            }
            pass4_ok = {
                name: data.pass_at_k_by_parent.get(parent, False)
                for name, data in loaded.items()
            }
            pass1_ok = {
                name: data.pass_at_1_by_parent.get(parent, False)
                for name, data in loaded.items()
            }
            all_selected_wrong = all(not value for value in selected_ok.values())
            if all(pass4_ok.values()) and all_selected_wrong:
                buckets["pass4_correct_selected_wrong"].append(self._case_payload(parent, exp_names))
            if all(pass1_ok.values()) and all_selected_wrong:
                buckets["pass1_correct_selected_wrong"].append(self._case_payload(parent, exp_names))
            if len(exp_names) == 2:
                a, b = exp_names
                if not selected_ok[a] and selected_ok[b]:
                    buckets["a_wrong_b_correct"].append(self._case_payload(parent, exp_names))
                if selected_ok[a] and not selected_ok[b]:
                    buckets["a_correct_b_wrong"].append(self._case_payload(parent, exp_names))

            if any(not ok for ok in selected_ok.values()):
                payload = self._case_payload(parent, exp_names)
                best_wrong_score = (-math.inf, -math.inf, -math.inf, -math.inf)
                for name, data in loaded.items():
                    info = self._selected_info(data, parent)
                    if info["is_correct"]:
                        continue
                    score_row = data.scores_by_id.get(str(info.get("record_id") or ""), {})
                    best_wrong_score = max(best_wrong_score, _score_rank_key(score_row))
                payload["_rank"] = best_wrong_score
                buckets["high_score_wrong"].append(payload)

        raw_counts = {key: len(value) for key, value in buckets.items()}
        for key, rows in buckets.items():
            if key == "high_score_wrong":
                rows.sort(key=lambda row: row.get("_rank", ()), reverse=True)
                for row in rows:
                    row.pop("_rank", None)
            else:
                rows.sort(key=lambda row: row["parent_id"])
            buckets[key] = rows[:limit]

        return {
            "exp_names": exp_names,
            "counts": raw_counts,
            "cases": buckets,
        }

    def _rollout_detail_payload(
        self,
        exp_name: str,
        data: ExperimentData,
        row: JsonDict,
        selected_rollout_id: Any,
    ) -> JsonDict:
        rid = str(row.get("id") or "")
        ev = data.eval_by_rollout.get(rid, {})
        rec = self._load_stage3_records(exp_name).get(rid)
        rollout = _safe_int(row.get("rollout_id"))
        return {
            "id": rid,
            "rollout_id": rollout,
            "selected": selected_rollout_id == rollout,
            "stage3_found": isinstance(rec, dict),
            "score": row,
            "math_verify": ev,
            "stage3": rec if isinstance(rec, dict) else None,
        }

    def _rollout_compare_links_payload(self, exp_names: List[str], parent_id: str) -> List[JsonDict]:
        if len(exp_names) != 2:
            return []
        stage1_by_exp = {name: self._load_stage1_records(name) for name in exp_names}
        stage3_by_exp = {name: self._load_stage3_records(name) for name in exp_names}
        links: List[JsonDict] = []
        for rollout_id in range(1, 5):
            record_ids = {name: _record_id_for(parent_id, rollout_id) for name in exp_names}
            missing_stage1 = [
                name for name in exp_names if record_ids[name] not in stage1_by_exp[name]
            ]
            missing_stage3 = [
                name for name in exp_names if record_ids[name] not in stage3_by_exp[name]
            ]
            stage1_equal = False
            if not missing_stage1:
                signatures = [
                    _stage1_signature(stage1_by_exp[name][record_ids[name]])
                    for name in exp_names
                ]
                stage1_equal = signatures[0] == signatures[1]
            links.append(
                {
                    "rollout_id": rollout_id,
                    "record_ids": record_ids,
                    "stage1_equal": stage1_equal,
                    "stage1_missing": missing_stage1,
                    "stage3_missing": missing_stage3,
                    "clickable": stage1_equal and not missing_stage3,
                }
            )
        return links

    def _copy_cell_payload(
        self,
        exp_name: str,
        rollout_id: int,
        record_id: str,
        stage: str,
        record: Optional[JsonDict],
    ) -> JsonDict:
        return {
            "experiment": exp_name,
            "rollout_id": rollout_id,
            "record_id": record_id,
            "stage": stage,
            "found": isinstance(record, dict),
            "record": record if isinstance(record, dict) else None,
        }

    def _rollout_copy_rows_payload(self, exp_names: List[str], parent_id: str) -> List[JsonDict]:
        rows: List[JsonDict] = []
        for name in exp_names:
            data = self._load_data(name)
            rollout_rows = sorted(
                data.scores_by_parent.get(parent_id) or [],
                key=lambda item: _safe_int(item.get("rollout_id")),
            )
            stage1_records = self._load_stage1_records(name)
            stage2_records = self._load_stage2_records(name)
            stage3_records = self._load_stage3_records(name)
            for row in rollout_rows:
                rollout_id = _safe_int(row.get("rollout_id"))
                record_id = str(row.get("id") or _record_id_for(parent_id, rollout_id))
                math_verify = data.eval_by_rollout.get(record_id, {})
                stage1 = stage1_records.get(record_id)
                stage2 = stage2_records.get(record_id)
                stage3 = stage3_records.get(record_id)
                all_info = {
                    "experiment": name,
                    "parent_id": parent_id,
                    "rollout_id": rollout_id,
                    "record_id": record_id,
                    "score": row,
                    "math_verify": math_verify,
                    "stage1": stage1,
                    "stage2": stage2,
                    "stage3": stage3,
                }
                rows.append(
                    {
                        "experiment": name,
                        "rollout_id": rollout_id,
                        "record_id": record_id,
                        "cells": {
                            "stage1": self._copy_cell_payload(name, rollout_id, record_id, "stage1", stage1),
                            "stage2": self._copy_cell_payload(name, rollout_id, record_id, "stage2", stage2),
                            "stage3": self._copy_cell_payload(name, rollout_id, record_id, "stage3", stage3),
                            "all": all_info,
                        },
                    }
                )
        return rows

    def problem_payload(self, exp_names: List[str], parent_id: str) -> JsonDict:
        exp_names = self._normalize_exp_names(exp_names)
        parent_id = str(parent_id).strip()
        if not parent_id:
            raise ValueError("parent_id is empty")
        tables: Dict[str, JsonDict] = {}
        analysis_experiments: Dict[str, JsonDict] = {}
        question = ""
        gold = ""
        source = ""
        for name in exp_names:
            data = self._load_data(name)
            rows = data.scores_by_parent.get(parent_id)
            if not rows:
                raise KeyError(f"parent_id not found in {name}: {parent_id}")
            selected = self._selected_info(data, parent_id)
            table_rows: List[JsonDict] = []
            for rank, row in enumerate(rows, start=1):
                rid = str(row.get("id") or "")
                ev = data.eval_by_rollout.get(rid, {})
                rollout = _safe_int(row.get("rollout_id"))
                prove_required_nodes = row.get("prove_required_nodes")
                if prove_required_nodes in (None, ""):
                    prove_required_nodes = row.get("total_nodes", "")
                table_rows.append(
                    {
                        "rank": rank,
                        "id": rid,
                        "success_ratio": row.get("success_ratio", ""),
                        "prove_required_nodes": prove_required_nodes,
                        "prove_success_nodes": row.get("prove_success_nodes", ""),
                        "lean_pass_nodes": row.get("lean_pass_nodes", ""),
                        "rollout_id": rollout,
                        "math_verify_skipped": not bool(ev),
                        "math_verify_correct": _to_bool_correct(ev.get("is_correct")),
                        "selected": selected.get("rollout_id") == rollout,
                    }
                )
            rollout_details = [
                self._rollout_detail_payload(name, data, row, selected.get("rollout_id"))
                for row in sorted(rows, key=lambda item: _safe_int(item.get("rollout_id")))
            ]
            tables[name] = {
                "selected": selected,
                "pass_at_1": data.pass_at_1_by_parent.get(parent_id, False),
                "pass_at_k": data.pass_at_k_by_parent.get(parent_id, False),
                "rows": table_rows,
                "rollouts": rollout_details,
            }
            analysis_experiments[name] = {
                "selected": selected,
                "pass_at_1": data.pass_at_1_by_parent.get(parent_id, False),
                "pass_at_k": data.pass_at_k_by_parent.get(parent_id, False),
                "rollouts": rollout_details,
            }
            question = question or str(rows[0].get("question") or "")
            gold = gold or str(rows[0].get("gold") or "")
            source = source or data.parent_source.get(parent_id, "")
        return {
            "parent_id": parent_id,
            "source": source,
            "question": question,
            "gold": gold,
            "exp_names": exp_names,
            "tables": tables,
            "rollout_compare_links": self._rollout_compare_links_payload(exp_names, parent_id),
            "rollout_copy_rows": self._rollout_copy_rows_payload(exp_names, parent_id),
            "analysis": {
                "parent_id": parent_id,
                "source": source,
                "question": question,
                "gold": gold,
                "experiments": analysis_experiments,
            },
        }

    def render_record_html(self, exp_name: str, record_id: str) -> str:
        exp_name = str(exp_name).strip()
        record_id = str(record_id).strip()
        if not exp_name:
            raise ValueError("exp_name is empty")
        if not record_id:
            raise ValueError("record_id is empty")
        rec = self._load_stage3_records(exp_name).get(record_id)
        if rec is None:
            raise KeyError(f"record_id not found in {exp_name}: {record_id}")
        nodes = _extract_nodes(rec, self.source)
        graph, node_info = build_dag(nodes)
        out_path = self.cache_dir / f"{exp_name}__{record_id}.html"
        with self._cache_lock:
            if self.graph_only:
                create_interactive_graph_only_visualization(
                    G=graph,
                    node_info=node_info,
                    title=f"Stage3 DAG ({record_id})",
                    subtitle=f"experiment: {exp_name}",
                    filename=str(out_path),
                )
            else:
                create_interactive_visualization(
                    G=graph,
                    node_info=node_info,
                    proof_str=_build_proof_str(rec),
                    filename=str(out_path),
                )
        return out_path.read_text(encoding="utf-8")

    def graph_record_payload(self, exp_name: str, record_id: str) -> JsonDict:
        exp_name = str(exp_name).strip()
        record_id = str(record_id).strip()
        if not exp_name:
            raise ValueError("exp_name is empty")
        if not record_id:
            raise ValueError("record_id is empty")
        rec = self._load_stage3_records(exp_name).get(record_id)
        if rec is None:
            raise KeyError(f"record_id not found in {exp_name}: {record_id}")
        data = self._load_data(exp_name)
        score = data.scores_by_id.get(record_id, {})
        eval_row = data.eval_by_rollout.get(record_id, {})
        parent_id = _parent_id({"id": record_id})
        selected = data.selected_eval_by_parent.get(parent_id, {})
        return {
            "experiment": exp_name,
            "record_id": record_id,
            "parent_id": parent_id,
            "source": score.get("source") or eval_row.get("source") or data.parent_source.get(parent_id, ""),
            "question": score.get("question", ""),
            "gold": score.get("gold") or eval_row.get("gold", ""),
            "rollout_id": _rollout_id({"id": record_id}),
            "response": score.get("response", ""),
            "score": score,
            "math_verify": eval_row,
            "selected_by_step_proof": _rollout_id(selected) == _rollout_id({"id": record_id}),
            "stage3": rec,
        }

    def render_compare_html(
        self,
        exp_names: List[str],
        record_id: str,
        compare_fields: List[str],
    ) -> str:
        exp_names = self._normalize_exp_names(exp_names)
        compare_fields = [field for field in compare_fields if field]
        if not compare_fields:
            raise ValueError("compare_fields is empty")
        record_id = str(record_id).strip()
        if not record_id:
            raise ValueError("record_id is empty")

        base_rec = self._load_stage3_records(exp_names[0]).get(record_id)
        if base_rec is None:
            raise KeyError(f"record_id not found in {exp_names[0]}: {record_id}")
        nodes = _extract_nodes(base_rec, self.source)
        graph, node_info = build_dag(nodes)
        compare_payload: Dict[str, Dict[str, Dict[str, Any]]] = {
            node_id: {} for node_id in node_info.keys()
        }
        for exp_name in exp_names:
            rec = self._load_stage3_records(exp_name).get(record_id)
            if rec is None:
                raise KeyError(f"record_id not found in {exp_name}: {record_id}")
            node_map = {str(node.get("id", "")).strip(): node for node in _extract_nodes(rec, self.source)}
            for node_id in compare_payload:
                node_payload = node_map.get(node_id, {})
                compare_payload[node_id][exp_name] = {
                    field_name: _get_nested_value(node_payload, field_name)
                    for field_name in compare_fields
                }
        out_path = self.cache_dir / f"compare__{'__'.join(exp_names)}__{record_id}.html"
        with self._cache_lock:
            create_interactive_visualization(
                G=graph,
                node_info=node_info,
                proof_str=_build_proof_str(base_rec),
                filename=str(out_path),
                compare_payload=compare_payload,
                compare_fields=compare_fields,
                compare_experiments=exp_names,
            )
        return out_path.read_text(encoding="utf-8")

    def _assert_rollout_stage1_equal(
        self,
        exp_names: List[str],
        parent_id: str,
        rollout_id: int,
    ) -> Dict[str, str]:
        if len(exp_names) != 2:
            raise ValueError("rollout pair comparison requires exactly 2 experiments")
        record_ids = {name: _record_id_for(parent_id, rollout_id) for name in exp_names}
        stage1_by_exp = {name: self._load_stage1_records(name) for name in exp_names}
        stage3_by_exp = {name: self._load_stage3_records(name) for name in exp_names}
        for name in exp_names:
            rid = record_ids[name]
            if rid not in stage1_by_exp[name]:
                raise KeyError(f"stage1 record not found in {name}: {rid}")
            if rid not in stage3_by_exp[name]:
                raise KeyError(f"stage3 record not found in {name}: {rid}")
        signatures = [
            _stage1_signature(stage1_by_exp[name][record_ids[name]])
            for name in exp_names
        ]
        if signatures[0] != signatures[1]:
            raise ValueError(
                f"stage1 graph differs for rollout {rollout_id}: {exp_names[0]} vs {exp_names[1]}"
            )
        return record_ids

    def _vis_graph_payload(
        self,
        graph: Any,
        node_info: Dict[str, JsonDict],
    ) -> JsonDict:
        nodes_data: List[JsonDict] = []
        edges_data: List[JsonDict] = []
        for node in sorted([str(item) for item in graph.nodes()], key=_node_sort_key):
            info = node_info.get(node, {})
            node_type = info.get("type", "unknown")
            skipped = (
                info.get("form_status") == "skipped"
                or info.get("prove_status") == "skipped"
                or (info.get("formalization") or {}).get("skipped") is True
                or (info.get("solved_lemma") or {}).get("skipped") is True
            )
            border = "#d12f2f"
            if skipped:
                border = "#111827"
            elif (info.get("solved_lemma") or {}).get("lean_verify", False):
                border = "#0f9f6e"
            elif (info.get("formalization") or {}).get("lean_pass", False):
                border = "#f59e0b"

            if node_type == "given":
                color, shape, size = "#e8e4dc", "box", 28
            elif node_type == "introduced":
                color, shape, size = "#e8dff5", "diamond", 30
            elif node_type == "derived":
                color, shape, size = "#d7e6f5", "dot", 26
            elif node_type == "answer":
                color, shape, size = "#a3c2a8", "star", 38
            elif node_type == "solution":
                color, shape, size = "#a3c2a8", "star", 38
            else:
                color, shape, size = "#d7e6f5", "dot", 26

            nodes_data.append(
                {
                    "id": node,
                    "label": node,
                    "shape": shape,
                    "size": size,
                    "color": {
                        "background": color,
                        "border": border,
                        "highlight": {"background": color, "border": border},
                    },
                    "borderWidth": 3,
                    "font": {"size": 14, "color": "#111827"},
                    "chosen": False,
                }
            )
        for src, dst in graph.edges():
            edges_data.append(
                {
                    "from": str(src),
                    "to": str(dst),
                    "arrows": "to",
                    "color": {"color": "#64748b", "highlight": "#334155", "hover": "#334155"},
                    "width": 2,
                }
            )
        return {"nodes": nodes_data, "edges": edges_data}

    def render_rollout_pair_compare_html(
        self,
        exp_names: List[str],
        parent_id: str,
        rollout_id: int,
    ) -> str:
        exp_names = self._normalize_exp_names(exp_names)
        parent_id = str(parent_id).strip()
        if not parent_id:
            raise ValueError("parent_id is empty")
        rollout_id = _safe_int(rollout_id)
        if rollout_id < 1:
            raise ValueError("rollout_id is invalid")
        record_ids = self._assert_rollout_stage1_equal(exp_names, parent_id, rollout_id)

        stage1_records = {name: self._load_stage1_records(name)[record_ids[name]] for name in exp_names}
        stage3_records = {name: self._load_stage3_records(name)[record_ids[name]] for name in exp_names}
        stage1_nodes = _extract_nodes(stage1_records[exp_names[0]], "graph")
        base_graph, stage1_info = build_dag(stage1_nodes)

        node_info_by_exp: Dict[str, Dict[str, JsonDict]] = {}
        graph_payloads: Dict[str, JsonDict] = {}
        for name in exp_names:
            stage3_nodes = _extract_nodes(stage3_records[name], self.source)
            _, stage3_info = build_dag(stage3_nodes)
            merged_info = {
                node_id: dict(stage3_info.get(node_id) or stage1_info.get(node_id) or {})
                for node_id in stage1_info.keys()
            }
            node_info_by_exp[name] = merged_info
            graph_payloads[name] = self._vis_graph_payload(base_graph, merged_info)

        field_paths = [value for value, _ in COMPARE_FIELD_OPTIONS]
        field_labels = {field: _field_label(field) for field in field_paths}
        compare_payload: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for node_id in sorted(stage1_info.keys(), key=_node_sort_key):
            compare_payload[node_id] = {}
            for field in field_paths:
                compare_payload[node_id][field] = {
                    name: _get_nested_value(node_info_by_exp[name].get(node_id, {}), field)
                    for name in exp_names
                }

        title = f"Rollout {rollout_id} comparison"
        subtitle = f"{parent_id} / {exp_names[0]} vs {exp_names[1]}"
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; overflow: hidden; }}
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2937; background: #f5f6f8; }}
    .page {{ display: grid; grid-template-columns: minmax(520px, 58vw) 1fr; height: 100vh; min-height: 0; overflow: hidden; }}
    .graphs {{ display: grid; grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; padding: 8px; min-width: 0; min-height: 0; height: 100vh; overflow: hidden; }}
    .graph-card {{ min-height: 0; background: #fff; border: 1px solid #d8dde5; border-radius: 8px; overflow: hidden; display: grid; grid-template-rows: auto minmax(0, 1fr); }}
    .graph-head {{ display: flex; justify-content: space-between; gap: 8px; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; font-size: 13px; }}
    .graph-head strong {{ word-break: break-all; }}
    .network {{ width: 100%; height: 100%; min-height: 0; overflow: hidden; }}
    .side {{ min-width: 0; min-height: 0; border-left: 1px solid #d8dde5; background: #fff; display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }}
    .title {{ padding: 12px 14px; border-bottom: 1px solid #d8dde5; }}
    .title h1 {{ margin: 0 0 4px; font-size: 18px; }}
    .title .muted {{ color: #667085; font-size: 12px; word-break: break-all; }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: #475569; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
    .swatch {{ width: 12px; height: 12px; border: 2px solid #111827; border-radius: 3px; display: inline-block; }}
    #detail {{ overflow: auto; padding: 12px 14px 24px; }}
    .placeholder {{ color: #667085; font-size: 13px; padding: 12px; }}
    .node-title {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 10px; }}
    .node-title h2 {{ margin: 0; font-size: 17px; }}
    .field-card {{ border: 1px solid #d8dde5; border-radius: 8px; overflow: hidden; margin-bottom: 10px; background: #fbfcfe; }}
    .field-title {{ padding: 8px 10px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; font-weight: 700; font-size: 13px; }}
    .field-values {{ display: grid; grid-template-columns: 1fr 1fr; }}
    .exp-value {{ min-width: 0; padding: 8px 10px; border-right: 1px solid #e5e7eb; }}
    .exp-value:last-child {{ border-right: 0; }}
    .exp-name {{ color: #475569; font-size: 12px; font-weight: 700; margin-bottom: 6px; word-break: break-all; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }}
    .selected-note {{ color: #2563eb; font-size: 12px; }}
    @media (max-width: 1000px) {{
      .page {{ grid-template-columns: 1fr; grid-template-rows: 62vh 1fr; }}
      .side {{ border-left: 0; border-top: 1px solid #d8dde5; }}
      .field-values {{ grid-template-columns: 1fr; }}
      .exp-value {{ border-right: 0; border-bottom: 1px solid #e5e7eb; }}
      .exp-value:last-child {{ border-bottom: 0; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="graphs">
      <section class="graph-card">
        <div class="graph-head"><strong id="expAName"></strong><span class="selected-note">top graph</span></div>
        <div id="graphA" class="network"></div>
      </section>
      <section class="graph-card">
        <div class="graph-head"><strong id="expBName"></strong><span class="selected-note">bottom graph</span></div>
        <div id="graphB" class="network"></div>
      </section>
    </div>
    <aside class="side">
      <div class="title">
        <h1>{html.escape(title)}</h1>
        <div class="muted">{html.escape(subtitle)}</div>
        <div class="legend">
          <span><i class="swatch" style="border-color:#0f9f6e"></i>prove ok</span>
          <span><i class="swatch" style="border-color:#f59e0b"></i>form ok</span>
          <span><i class="swatch" style="border-color:#d12f2f"></i>failed</span>
          <span><i class="swatch" style="border-color:#111827"></i>skipped</span>
        </div>
      </div>
      <div id="detail"><div class="placeholder">Click a node in either graph to compare fields.</div></div>
    </aside>
  </div>
  <script>
    const expNames = {json.dumps(exp_names, ensure_ascii=False)};
    const graphPayloads = {json.dumps(graph_payloads, ensure_ascii=False)};
    const compareData = {json.dumps(compare_payload, ensure_ascii=False)};
    const fieldPaths = {json.dumps(field_paths, ensure_ascii=False)};
    const fieldLabels = {json.dumps(field_labels, ensure_ascii=False)};
    const recordIds = {json.dumps(record_ids, ensure_ascii=False)};
    const networks = [];

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function valueText(value) {{
      if (value === null || value === undefined || value === "") return "N/A";
      if (typeof value === "object") return JSON.stringify(value, null, 2);
      return String(value);
    }}

    function renderNodeCompare(nodeId) {{
      for (const network of networks) {{
        network.selectNodes([nodeId]);
      }}
      let html = `<div class="node-title"><h2>${{escapeHtml(nodeId)}}</h2><span class="selected-note">same stage1 node</span></div>`;
      const nodeCompare = compareData[nodeId] || {{}};
      for (const field of fieldPaths) {{
        html += `<section class="field-card"><div class="field-title">${{escapeHtml(fieldLabels[field] || field)}}</div><div class="field-values">`;
        for (const expName of expNames) {{
          const values = nodeCompare[field] || {{}};
          html += `<div class="exp-value">
            <div class="exp-name">${{escapeHtml(expName)}}</div>
            <pre>${{escapeHtml(valueText(values[expName]))}}</pre>
          </div>`;
        }}
        html += `</div></section>`;
      }}
      document.getElementById("detail").innerHTML = html;
    }}

    function makeNetwork(elId, expName) {{
      const payload = graphPayloads[expName] || {{nodes: [], edges: []}};
      const network = new vis.Network(
        document.getElementById(elId),
        {{
          nodes: new vis.DataSet(payload.nodes || []),
          edges: new vis.DataSet(payload.edges || []),
        }},
        {{
          physics: {{
            enabled: true,
            solver: "hierarchicalRepulsion",
            hierarchicalRepulsion: {{
              centralGravity: 0.0,
              springLength: 200,
              springConstant: 0.01,
              nodeDistance: 150,
              damping: 0.09
            }}
          }},
          interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
          edges: {{
            smooth: {{ type: "continuous", forceDirection: "none" }},
            color: {{ color: "#64748b", highlight: "#334155", hover: "#334155" }},
            width: 2
          }},
          nodes: {{ chosen: false }},
        }}
      );
      network.on("click", (params) => {{
        if (params.nodes && params.nodes.length > 0) renderNodeCompare(params.nodes[0]);
      }});
      network.once("stabilized", () => network.fit({{ animation: false }}));
      networks.push(network);
      return network;
    }}

    document.getElementById("expAName").textContent = `${{expNames[0]}} / ${{recordIds[expNames[0]] || ""}}`;
    document.getElementById("expBName").textContent = `${{expNames[1]}} / ${{recordIds[expNames[1]] || ""}}`;
    makeNetwork("graphA", expNames[0]);
    makeNetwork("graphB", expNames[1]);
  </script>
</body>
</html>"""

    def _normalize_exp_names(self, exp_names: Iterable[str]) -> List[str]:
        names = [str(name).strip() for name in exp_names if str(name).strip()]
        if not names:
            raise ValueError("select 1 or 2 experiments")
        if len(names) > 2:
            raise ValueError("at most 2 experiments can be selected")
        complete = set(self.complete_experiment_names())
        for name in names:
            if name not in complete:
                raise ValueError(f"experiment is not complete or not found: {name}")
        return names


def _html_page() -> str:
    fields_json = json.dumps(
        [{"value": value, "label": label} for value, label in COMPARE_FIELD_OPTIONS],
        ensure_ascii=False,
    )
    return HTML_PAGE.replace("__COMPARE_FIELDS__", fields_json)


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Step Proof Compare Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --line: #d8dde5;
      --text: #1f2933;
      --muted: #667085;
      --blue: #2563eb;
      --green: #11845b;
      --red: #c43d3d;
      --amber: #9a6700;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }
    main { padding: 14px; min-width: 0; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 15px; margin: 14px 0 8px; }
    label { font-size: 13px; }
    input[type="text"] {
      width: 100%;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    button {
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
    }
    button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .muted { color: var(--muted); font-size: 12px; }
    .error { color: var(--red); white-space: pre-wrap; font-size: 13px; }
    .exp-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      margin: 8px 0;
      background: #fff;
    }
    .exp-card.incomplete { opacity: 0.7; background: #fafafa; }
    .exp-name { font-weight: 700; word-break: break-all; }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 8px;
      margin-top: 6px;
      font-size: 12px;
    }
    .tabs { display: flex; gap: 6px; margin-bottom: 10px; border-bottom: 1px solid var(--line); }
    .tab {
      border: 0;
      border-radius: 6px 6px 0 0;
      padding: 9px 12px;
      background: transparent;
    }
    .tab.active { background: var(--panel); border: 1px solid var(--line); border-bottom-color: var(--panel); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }
    th { background: #f8fafc; position: sticky; top: 0; z-index: 1; }
    .table-wrap { overflow: auto; max-height: 58vh; border: 1px solid var(--line); border-radius: 8px; }
    a { color: var(--blue); text-decoration: none; cursor: pointer; }
    .ok { color: var(--green); font-weight: 700; }
    .bad { color: var(--red); font-weight: 700; }
    .warn { color: var(--amber); }
    .case-grid { display: grid; gap: 12px; }
    .case-title { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
    .pill {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      white-space: nowrap;
    }
    .problem-meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
      font-size: 13px;
    }
    .question {
      white-space: pre-wrap;
      line-height: 1.45;
      max-height: 160px;
      overflow: auto;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .selected-row { background: #f0fdf4; }
    .rollout-summary { margin-top: 12px; }
    .rollout-summary h3 { margin: 14px 0 8px; font-size: 15px; }
    .rollout-list { display: grid; gap: 10px; margin-top: 12px; }
    .rollout-detail {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .rollout-head, .copy-head, .fact-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .copy-block {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 8px;
      background: #fbfcfe;
    }
    .copy-head {
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      font-size: 12px;
      font-weight: 700;
    }
    .copy-head button { padding: 4px 8px; font-size: 12px; }
    .copy-block pre {
      margin: 0;
      padding: 10px;
      max-height: 240px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    }
    .json-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 12px;
      background: #fbfcfe;
    }
    .json-box textarea {
      width: 100%;
      min-height: 62vh;
      resize: vertical;
      border: 0;
      outline: 0;
      padding: 12px;
      background: #fbfcfe;
      white-space: pre;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    }
    .graph-links {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .graph-link-row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      font-size: 13px;
    }
    .rollout-compare-links {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      font-size: 13px;
    }
    .rollout-compare-links .disabled-link {
      color: var(--muted);
      cursor: default;
    }
    .copy-matrix {
      margin-top: 12px;
    }
    .copy-matrix button {
      width: 100%;
      white-space: nowrap;
    }
    .fact-detail {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      margin-top: 8px;
      background: #fff;
    }
    .fact-detail summary { cursor: pointer; }
    .fact-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 1fr));
      gap: 8px;
    }
    #viewer { width: 100%; height: 76vh; border: 0; background: #fff; border-radius: 8px; }
    .field-list label { display: inline-flex; gap: 4px; align-items: center; margin: 2px 8px 2px 0; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .problem-meta { grid-template-columns: 1fr 1fr; }
      .fact-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Step Proof Compare</h1>
      <div class="muted">Root</div>
      <input id="rootPath" type="text" />
      <div class="row" style="margin-top:8px;">
        <button id="scanBtn" class="primary">Scan</button>
        <button id="mineBtn">Mine cases</button>
      </div>
      <div id="sideError" class="error" style="margin-top:8px;"></div>
      <h2>Complete experiments</h2>
      <div id="experiments"></div>
      <details style="margin-top:10px;">
        <summary class="muted">Incomplete diagnostics</summary>
        <div id="diagnostics"></div>
      </details>
      <h2>Problem lookup</h2>
      <input id="parentInput" type="text" placeholder="aime25_test__0" />
      <div class="row" style="margin-top:8px;">
        <button id="openProblemBtn">Open problem</button>
      </div>
      <h2>Graph lookup</h2>
      <input id="recordInput" type="text" placeholder="aime25_test__0__rollout_2" />
      <div class="row" style="margin-top:8px;">
        <button id="renderBtn">Render graph</button>
        <button id="compareGraphBtn">Compare graph</button>
      </div>
      <div class="field-list" id="fieldOptions" style="margin-top:8px;"></div>
    </aside>
    <main>
      <div class="tabs">
        <button class="tab active" data-tab="experimentsTab">Experiments</button>
        <button class="tab" data-tab="casesTab">Cases</button>
        <button class="tab" data-tab="problemTab">Problem</button>
        <button class="tab" data-tab="graphTab">Graph</button>
      </div>
      <section id="experimentsTab" class="tab-panel active">
        <div class="panel">
          <h2>Experiment summary</h2>
          <div id="summaryTable"></div>
        </div>
      </section>
      <section id="casesTab" class="tab-panel">
        <div class="panel">
          <div class="case-title">
            <h2>Mined cases</h2>
            <span id="caseStatus" class="muted"></span>
          </div>
          <div id="cases" class="case-grid"></div>
        </div>
      </section>
      <section id="problemTab" class="tab-panel">
        <div class="panel">
          <div class="case-title">
            <h2>Problem detail</h2>
            <span id="problemStatus" class="muted"></span>
          </div>
          <div id="problem"></div>
        </div>
      </section>
      <section id="graphTab" class="tab-panel">
        <div class="panel">
          <div class="case-title">
            <h2>Graph visualization</h2>
            <div class="row">
              <button id="copyGraphJsonBtn" disabled>Copy current JSON</button>
              <span id="graphStatus" class="muted"></span>
            </div>
          </div>
          <iframe id="viewer" title="stage3 graph"></iframe>
        </div>
      </section>
    </main>
  </div>

  <script>
    const compareFields = __COMPARE_FIELDS__;
    let currentRoot = "";
    let completeNames = [];
    let copyBlockCounter = 0;
    let currentGraphJson = "";
    let currentRolloutCopyRows = [];

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function escapeAttr(value) {
      return escapeHtml(value).replaceAll("'", "&#39;");
    }

    async function copyText(text) {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
      }
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.focus();
      area.select();
      document.execCommand("copy");
      area.remove();
    }

    async function copyBlock(id, button) {
      const el = document.getElementById(id);
      if (!el) return;
      const label = button.textContent;
      try {
        await copyText(el.value ?? el.textContent ?? "");
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = label; }, 900);
      } catch (e) {
        button.textContent = "Failed";
        setTimeout(() => { button.textContent = label; }, 1200);
      }
    }

    function renderCopyBlock(title, value) {
      const id = `copyBlock${copyBlockCounter++}`;
      const text = String(value ?? "");
      return `<div class="copy-block">
        <div class="copy-head">
          <span>${escapeHtml(title)}</span>
          <button onclick="copyBlock('${id}', this)">Copy</button>
        </div>
        <pre id="${id}">${escapeHtml(text)}</pre>
      </div>`;
    }

    function renderJsonBox(title, value) {
      const id = `copyBlock${copyBlockCounter++}`;
      return `<div class="json-box">
        <div class="copy-head">
          <span>${escapeHtml(title)}</span>
          <button onclick="copyBlock('${id}', this)">Copy all</button>
        </div>
        <textarea id="${id}" readonly spellcheck="false">${escapeHtml(value)}</textarea>
      </div>`;
    }

    function renderGraphLinks(data) {
      const experiments = (data.analysis || {}).experiments || {};
      const rows = Object.entries(experiments).map(([name, exp]) => {
        const links = (exp.rollouts || []).map((rollout) => {
          const label = `rollout ${rollout.rollout_id}`;
          if (!rollout.stage3_found) return `<span class="muted">${escapeHtml(label)} missing</span>`;
          return `<a onclick="renderGraph('${escapeAttr(name)}', '${escapeAttr(rollout.id)}')">${escapeHtml(label)}</a>`;
        }).join("");
        return `<div class="graph-link-row">
          <strong>${escapeHtml(name)}</strong>
          ${links || `<span class="muted">No graph links.</span>`}
        </div>`;
      });
      if (!rows.length) return "";
      return `<div class="graph-links">${rows.join("")}</div>`;
    }

    function rolloutCompareReason(link) {
      if ((link.stage1_missing || []).length) return `stage1 missing: ${(link.stage1_missing || []).join(", ")}`;
      if (!link.stage1_equal) return "stage1 graph differs";
      if ((link.stage3_missing || []).length) return `stage3 missing: ${(link.stage3_missing || []).join(", ")}`;
      return "";
    }

    function renderRolloutCompareLinks(data) {
      const names = data.exp_names || [];
      const links = data.rollout_compare_links || [];
      if (names.length !== 2 || !links.length) return "";
      const pieces = links.map((link) => {
        const label = `rollout${link.rollout_id}`;
        if (link.clickable) {
          return `<a onclick="openRolloutCompare('${escapeAttr(data.parent_id)}', ${Number(link.rollout_id)}, '${escapeAttr(names.join(","))}')">${escapeHtml(label)}</a>`;
        }
        const reason = rolloutCompareReason(link);
        return `<span class="disabled-link" title="${escapeAttr(reason)}">${escapeHtml(label)}</span>`;
      }).join("");
      return `<div class="rollout-compare-links">
        <strong>Compare same rollout</strong>
        ${pieces}
        <span class="muted">enabled only when complete stage1 graphs match</span>
      </div>`;
    }

    async function copyRolloutInfo(rowIdx, key, button) {
      const row = currentRolloutCopyRows[rowIdx] || {};
      const cells = row.cells || {};
      const payload = cells[key] || {};
      const label = button.textContent;
      try {
        await copyText(JSON.stringify(payload, null, 2));
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = label; }, 900);
      } catch (e) {
        button.textContent = "Failed";
        setTimeout(() => { button.textContent = label; }, 1200);
      }
    }

    function renderRolloutCopyMatrix(data) {
      currentRolloutCopyRows = data.rollout_copy_rows || [];
      if (!currentRolloutCopyRows.length) return "";
      return `<div class="copy-matrix">
        <h2>Copy rollout stage info</h2>
        <div class="table-wrap"><table>
          <thead><tr>
            <th>Experiment</th>
            <th>Rollout</th>
            <th>stage1</th>
            <th>stage2</th>
            <th>stage3</th>
            <th>All info</th>
          </tr></thead>
          <tbody>${currentRolloutCopyRows.map((row, idx) => {
            const stage1 = ((row.cells || {}).stage1 || {}).found;
            const stage2 = ((row.cells || {}).stage2 || {}).found;
            const stage3 = ((row.cells || {}).stage3 || {}).found;
            return `<tr>
              <td>${escapeHtml(row.experiment)}</td>
              <td>${escapeHtml(row.rollout_id)}<br><span class="muted">${escapeHtml(row.record_id)}</span></td>
              <td><button onclick="copyRolloutInfo(${idx}, 'stage1', this)">Copy${stage1 ? "" : " missing"}</button></td>
              <td><button onclick="copyRolloutInfo(${idx}, 'stage2', this)">Copy${stage2 ? "" : " missing"}</button></td>
              <td><button onclick="copyRolloutInfo(${idx}, 'stage3', this)">Copy${stage3 ? "" : " missing"}</button></td>
              <td><button onclick="copyRolloutInfo(${idx}, 'all', this)">Copy all</button></td>
            </tr>`;
          }).join("")}</tbody>
        </table></div>
      </div>`;
    }

    function renderRolloutSummaryTables(data) {
      const tables = data.tables || {};
      const names = data.exp_names || Object.keys(tables);
      const blocks = names.map((name) => {
        const table = tables[name] || {};
        const rows = table.rows || [];
        if (!rows.length) {
          return `<div class="rollout-summary">
            <h3>${escapeHtml(name)}</h3>
            <div class="muted">No rollout rows.</div>
          </div>`;
        }
        return `<div class="rollout-summary">
          <h3>${escapeHtml(name)}</h3>
          <div class="table-wrap"><table>
            <thead><tr>
              <th>Rank</th>
              <th>Rollout</th>
              <th>Selected</th>
              <th>success_ratio</th>
              <th>prove_success_nodes</th>
              <th>prove_required_nodes</th>
              <th>Skip math verify</th>
              <th>Math verify</th>
            </tr></thead>
            <tbody>${rows.map((row) => `<tr class="${row.selected ? "selected-row" : ""}">
              <td>${escapeHtml(row.rank)}</td>
              <td><a onclick="renderGraph('${escapeAttr(name)}', '${escapeAttr(row.id)}')">${escapeHtml(row.rollout_id)}</a></td>
              <td>${boolMark(row.selected)}</td>
              <td>${escapeHtml(row.success_ratio)}</td>
              <td>${escapeHtml(row.prove_success_nodes)}</td>
              <td>${escapeHtml(row.prove_required_nodes)}</td>
              <td>${row.math_verify_skipped ? `<span class="warn">yes</span>` : `<span class="ok">no</span>`}</td>
              <td>${row.math_verify_skipped ? `<span class="muted">n/a</span>` : boolMark(row.math_verify_correct)}</td>
            </tr>`).join("")}</tbody>
          </table></div>
        </div>`;
      });
      return blocks.length ? `<h2>Rollouts</h2>${blocks.join("")}` : "";
    }

    function selectedExpNames() {
      return Array.from(document.querySelectorAll('input[name="exp"]:checked')).map((el) => el.value);
    }

    function selectedCompareFields() {
      return Array.from(document.querySelectorAll('input[name="compareField"]:checked')).map((el) => el.value);
    }

    function setTab(tabId) {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
    }

    async function getJSON(url, options) {
      const resp = await fetch(url, options);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function formBody(params) {
      return Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
    }

    function guardSelection() {
      const names = selectedExpNames();
      if (names.length < 1 || names.length > 2) {
        throw new Error("Please select 1 or 2 complete experiments.");
      }
      return names;
    }

    function renderFieldOptions() {
      const el = document.getElementById("fieldOptions");
      el.innerHTML = compareFields.map((item, idx) => `
        <label><input type="checkbox" name="compareField" value="${escapeHtml(item.value)}" ${idx < 2 ? "checked" : ""}/> ${escapeHtml(item.label)}</label>
      `).join("");
    }

    async function scanExperiments() {
      const root = document.getElementById("rootPath").value.trim();
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const data = await getJSON(`/api/experiments?root=${encodeURIComponent(root)}`);
        currentRoot = data.root;
        document.getElementById("rootPath").value = data.root;
        completeNames = data.complete_names || [];
        renderExperimentList(data.experiments || []);
        renderSummary(data.experiments || []);
        setTab("experimentsTab");
      } catch (e) {
        err.textContent = e.message || String(e);
      }
    }

    function openRolloutCompare(parentId, rolloutId, expNamesCsv) {
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const names = expNamesCsv ? expNamesCsv.split(",").filter(Boolean) : guardSelection();
        if (names.length !== 2) throw new Error("Please select exactly 2 experiments.");
        const params = new URLSearchParams({
          root: currentRoot,
          exp_names: names.join(","),
          parent_id: parentId,
          rollout_id: String(rolloutId),
        });
        window.open(`/rollout_compare?${params.toString()}`, "_blank", "noopener");
      } catch (e) {
        err.textContent = e.message || String(e);
      }
    }

    function renderExperimentList(experiments) {
      const target = document.getElementById("experiments");
      const complete = experiments.filter((exp) => exp.complete);
      if (!complete.length) {
        target.innerHTML = `<div class="error">No complete experiments found.</div>`;
      } else {
        target.innerHTML = complete.map((exp, idx) => {
          const s = exp.summary || {};
          return `<div class="exp-card">
            <label>
              <input type="checkbox" name="exp" value="${escapeHtml(exp.name)}" ${idx < 2 ? "checked" : ""}/>
              <span class="exp-name">${escapeHtml(exp.name)}</span>
            </label>
            <div class="metric-grid">
              <span>pass@1 ${escapeHtml(s.pass_at_1)}</span>
              <span>pass@4 ${escapeHtml(s.pass_at_4)}</span>
              <span>ours ${escapeHtml(s.ours)}</span>
              <span>hit ${escapeHtml(s.selection_hit_rate)}</span>
            </div>
          </div>`;
        }).join("");
      }
      document.querySelectorAll('input[name="exp"]').forEach((box) => {
        box.addEventListener("change", () => {
          const checked = selectedExpNames();
          if (checked.length > 2) box.checked = false;
        });
      });

      const diag = document.getElementById("diagnostics");
      const incomplete = experiments.filter((exp) => !exp.complete);
      diag.innerHTML = incomplete.length ? incomplete.map((exp) => `
        <div class="exp-card incomplete">
          <div class="exp-name">${escapeHtml(exp.name)}</div>
          <div class="muted">${escapeHtml((exp.missing || []).join(", "))}</div>
        </div>
      `).join("") : `<div class="muted">None</div>`;
    }

    function renderSummary(experiments) {
      const complete = experiments.filter((exp) => exp.complete);
      if (!complete.length) {
        document.getElementById("summaryTable").innerHTML = `<div class="muted">Scan a root with complete experiments.</div>`;
        return;
      }
      document.getElementById("summaryTable").innerHTML = `<div class="table-wrap"><table>
        <thead><tr><th>Experiment</th><th>Total</th><th>pass@1</th><th>pass@4</th><th>ours</th><th>selection hit</th></tr></thead>
        <tbody>${complete.map((exp) => {
          const s = exp.summary || {};
          return `<tr>
            <td>${escapeHtml(exp.name)}</td>
            <td>${escapeHtml(s.total)}</td>
            <td>${escapeHtml(s.pass_at_1)}</td>
            <td>${escapeHtml(s.pass_at_4)}</td>
            <td>${escapeHtml(s.ours)}</td>
            <td>${escapeHtml(s.selection_hit_rate)}</td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>`;
    }

    async function mineCases() {
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const names = guardSelection();
        document.getElementById("caseStatus").textContent = "loading...";
        const data = await getJSON("/api/mine_cases", {
          method: "POST",
          headers: {"Content-Type": "application/x-www-form-urlencoded"},
          body: formBody({root: currentRoot, exp_names: names.join(",")}),
        });
        renderCases(data);
        setTab("casesTab");
        document.getElementById("caseStatus").textContent = `${names.join(", ")}`;
      } catch (e) {
        err.textContent = e.message || String(e);
        document.getElementById("caseStatus").textContent = "";
      }
    }

    function boolMark(value) {
      return value ? `<span class="ok">yes</span>` : `<span class="bad">no</span>`;
    }

    function statusMark(value) {
      const text = String(value ?? "");
      const lower = text.toLowerCase();
      if (!text) return `<span class="muted">n/a</span>`;
      if (lower === "success" || lower === "true") return `<span class="ok">${escapeHtml(text)}</span>`;
      if (lower === "failed" || lower === "failure" || lower === "false" || lower === "error") return `<span class="bad">${escapeHtml(text)}</span>`;
      return `<span>${escapeHtml(text)}</span>`;
    }

    function renderFactDetail(fact, idx) {
      const parents = Array.isArray(fact.parent_fact_ids) ? fact.parent_fact_ids.join(", ") : String(fact.parent_fact_ids ?? "");
      const title = `${fact.fact_id || `fact_${idx + 1}`} ${fact.is_final_answer ? "(final)" : ""}`;
      const open = fact.is_final_answer ? "open" : "";
      return `<details class="fact-detail" ${open}>
        <summary>
          <span class="fact-head">
            <strong>${escapeHtml(title)}</strong>
            <span class="muted">origin ${escapeHtml(fact.origin)} / parents ${escapeHtml(parents)} / form ${statusMark(fact.form_status)} / prove ${statusMark(fact.prove_status)}</span>
          </span>
        </summary>
        ${renderCopyBlock("Fact text", fact.text)}
        ${renderCopyBlock("Proof obligation", fact.proof_obligation)}
        <div class="fact-grid">
          <div>
            <div class="muted" style="margin-top:8px;">formalization lean_pass ${statusMark(fact.formalization_lean_pass)}</div>
            ${renderCopyBlock("Form Lean", fact.formalization_lean_code)}
            ${renderCopyBlock("Form conversation", fact.formalization_conversation)}
            ${fact.formalization_error ? renderCopyBlock("Form error", fact.formalization_error) : ""}
          </div>
          <div>
            <div class="muted" style="margin-top:8px;">prove lean_pass ${statusMark(fact.solved_lemma_lean_pass)} / lean_verify ${statusMark(fact.solved_lemma_lean_verify)}</div>
            ${renderCopyBlock("Prove Lean", fact.solved_lemma_lean_code)}
            ${renderCopyBlock("Prove conversation", fact.prove_conversation)}
            ${fact.solved_lemma_error ? renderCopyBlock("Prove error", fact.solved_lemma_error) : ""}
          </div>
        </div>
      </details>`;
    }

    function renderRolloutDetail(rollout, expName) {
      const facts = rollout.facts || [];
      const title = `rollout ${escapeHtml(rollout.rollout_id)}${rollout.selected ? " / selected" : ""}`;
      return `<details class="rollout-detail" ${rollout.selected ? "open" : ""}>
        <summary>
          <span class="rollout-head">
            <strong>${title}</strong>
            <span class="muted">math_verify ${rollout.math_verify_correct ? "correct" : "wrong"} / score ${escapeHtml(rollout.success_ratio)} / proved ${escapeHtml(rollout.prove_success_nodes)}/${escapeHtml(rollout.prove_required_nodes)} / lean_pass ${escapeHtml(rollout.lean_pass_nodes)}</span>
          </span>
        </summary>
        <div class="row" style="margin-top:8px;">
          <a onclick="renderGraph('${escapeAttr(expName)}', '${escapeAttr(rollout.id)}')">view graph</a>
          <span class="muted">${rollout.stage3_found ? `${facts.length} stage3 fact(s)` : "stage3 record missing"}</span>
        </div>
        ${renderCopyBlock("Rollout result", rollout.answer)}
        ${facts.length ? facts.map(renderFactDetail).join("") : `<div class="muted" style="margin-top:8px;">No stage3 form/prove detail found.</div>`}
      </details>`;
    }

    function renderRolloutDetails(table, expName) {
      const rollouts = table.rollouts || [];
      if (!rollouts.length) return `<div class="muted">No rollout details.</div>`;
      return `<div class="rollout-list">${rollouts.map((rollout) => renderRolloutDetail(rollout, expName)).join("")}</div>`;
    }

    function renderCases(data) {
      const names = data.exp_names || [];
      const titles = {
        pass4_correct_selected_wrong: "1. pass@4 correct, selected wrong",
        pass1_correct_selected_wrong: "2. pass@1 correct, selected wrong",
        a_wrong_b_correct: "3. A wrong, B correct",
        a_correct_b_wrong: "4. A correct, B wrong",
        high_score_wrong: "5. High proof score but final answer wrong",
      };
      const target = document.getElementById("cases");
      target.innerHTML = Object.entries(titles).map(([key, title]) => {
        const rows = (data.cases || {})[key] || [];
        const count = (data.counts || {})[key] ?? rows.length;
        return `<div class="panel">
          <div class="case-title"><h2>${escapeHtml(title)}</h2><span class="pill">${escapeHtml(count)} total / ${rows.length} shown</span></div>
          ${renderCaseTable(rows, names)}
        </div>`;
      }).join("");
    }

    function renderCaseTable(rows, names) {
      if (!rows.length) return `<div class="muted">No cases.</div>`;
      const expHeads = names.map((name) => `<th>${escapeHtml(name)} selected</th><th>${escapeHtml(name)} correct rollouts</th>`).join("");
      return `<div class="table-wrap"><table>
        <thead><tr><th>Problem</th><th>Source</th><th>Gold</th>${expHeads}<th>Question</th></tr></thead>
        <tbody>${rows.map((row) => {
          const expCells = names.map((name) => {
            const info = (row.by_exp || {})[name] || {};
            const rollouts = ((row.correct_rollouts || {})[name] || []).join(", ");
            const selected = `${escapeHtml(info.rollout_id)} / ${boolMark(info.is_correct)} / score ${escapeHtml(info.success_ratio)}`;
            return `<td>${selected}</td><td>${escapeHtml(rollouts)}</td>`;
          }).join("");
          return `<tr>
            <td><a onclick="openProblem('${escapeHtml(row.parent_id)}')">${escapeHtml(row.parent_id)}</a></td>
            <td>${escapeHtml(row.source)}</td>
            <td>${escapeHtml(row.gold)}</td>
            ${expCells}
            <td>${escapeHtml(row.question)}</td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>`;
    }

    async function openProblem(parentId) {
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const names = guardSelection();
        const parent = parentId || document.getElementById("parentInput").value.trim();
        if (!parent) throw new Error("Please input a full problem id.");
        document.getElementById("parentInput").value = parent;
        document.getElementById("problemStatus").textContent = "loading...";
        const data = await getJSON(`/api/problem?root=${encodeURIComponent(currentRoot)}&exp_names=${encodeURIComponent(names.join(","))}&parent_id=${encodeURIComponent(parent)}`);
        renderProblem(data);
        setTab("problemTab");
        document.getElementById("problemStatus").textContent = parent;
      } catch (e) {
        err.textContent = e.message || String(e);
        document.getElementById("problemStatus").textContent = "";
      }
    }

    function renderProblem(data) {
      copyBlockCounter = 0;
      currentRolloutCopyRows = [];
      const names = data.exp_names || [];
      const meta = `<div class="problem-meta">
        <div><strong>ID</strong><br>${escapeHtml(data.parent_id)}</div>
        <div><strong>Source</strong><br>${escapeHtml(data.source)}</div>
        <div><strong>Gold</strong><br>${escapeHtml(data.gold)}</div>
        <div><strong>Experiments</strong><br>${escapeHtml(names.join(", "))}</div>
      </div>
      <h2>Question</h2>
      <div class="question">${escapeHtml(data.question)}</div>`;
      const analysisJson = JSON.stringify(data.analysis || {}, null, 2);
      document.getElementById("problem").innerHTML = meta + renderRolloutCompareLinks(data) + renderRolloutCopyMatrix(data) + renderRolloutSummaryTables(data) + renderGraphLinks(data) + renderJsonBox("All rollout / form / prove JSON", analysisJson);
    }

    async function renderGraph(expName, recordId) {
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const names = selectedExpNames();
        const exp = expName || names[0];
        const record = recordId || document.getElementById("recordInput").value.trim();
        if (!exp) throw new Error("Please select an experiment.");
        if (!record) throw new Error("Please input a full rollout id.");
        document.getElementById("recordInput").value = record;
        document.getElementById("graphStatus").textContent = "rendering...";
        const data = await getJSON("/api/render", {
          method: "POST",
          headers: {"Content-Type": "application/x-www-form-urlencoded"},
          body: formBody({root: currentRoot, exp_name: exp, record_id: record}),
        });
        document.getElementById("viewer").srcdoc = data.html;
        currentGraphJson = JSON.stringify(data.record || {}, null, 2);
        document.getElementById("copyGraphJsonBtn").disabled = !currentGraphJson;
        setTab("graphTab");
        document.getElementById("graphStatus").textContent = `${exp} / ${record}`;
      } catch (e) {
        err.textContent = e.message || String(e);
        document.getElementById("graphStatus").textContent = "";
      }
    }

    async function compareGraph() {
      const err = document.getElementById("sideError");
      err.textContent = "";
      try {
        const names = guardSelection();
        const record = document.getElementById("recordInput").value.trim();
        if (!record) throw new Error("Please input a full rollout id.");
        const fields = selectedCompareFields();
        if (!fields.length) throw new Error("Please select at least one compare field.");
        document.getElementById("graphStatus").textContent = "rendering comparison...";
        const data = await getJSON("/api/compare_graph", {
          method: "POST",
          headers: {"Content-Type": "application/x-www-form-urlencoded"},
          body: formBody({root: currentRoot, exp_names: names.join(","), record_id: record, compare_fields: fields.join(",")}),
        });
        document.getElementById("viewer").srcdoc = data.html;
        currentGraphJson = "";
        document.getElementById("copyGraphJsonBtn").disabled = true;
        setTab("graphTab");
        document.getElementById("graphStatus").textContent = `${names.join(", ")} / ${record}`;
      } catch (e) {
        err.textContent = e.message || String(e);
        document.getElementById("graphStatus").textContent = "";
      }
    }

    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setTab(tab.dataset.tab)));
    document.getElementById("scanBtn").addEventListener("click", scanExperiments);
    document.getElementById("mineBtn").addEventListener("click", mineCases);
    document.getElementById("openProblemBtn").addEventListener("click", () => openProblem(""));
    document.getElementById("renderBtn").addEventListener("click", () => renderGraph("", ""));
    document.getElementById("compareGraphBtn").addEventListener("click", compareGraph);
    document.getElementById("copyGraphJsonBtn").addEventListener("click", async (e) => {
      const button = e.currentTarget;
      const label = button.textContent;
      try {
        await copyText(currentGraphJson || "{}");
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = label; }, 900);
      } catch (err) {
        button.textContent = "Failed";
        setTimeout(() => { button.textContent = label; }, 1200);
      }
    });
    document.getElementById("parentInput").addEventListener("keydown", (e) => { if (e.key === "Enter") openProblem(""); });
    document.getElementById("recordInput").addEventListener("keydown", (e) => { if (e.key === "Enter") renderGraph("", ""); });

    renderFieldOptions();
    scanExperiments();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: JsonDict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_post_form(handler: BaseHTTPRequestHandler) -> Dict[str, List[str]]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("invalid content length")
    raw = handler.rfile.read(content_length).decode("utf-8")
    return parse_qs(raw, keep_blank_values=True)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def create_handler(app: StepProofCompareApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path == "/":
                    body = _html_page().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/experiments":
                    root = (query.get("root") or [""])[0].strip()
                    if root:
                        app.results_root = Path(root).expanduser().resolve()
                    app._status_cache = None
                    app._data_cache.clear()
                    app._stage1_cache.clear()
                    app._stage2_cache.clear()
                    app._stage3_cache.clear()
                    _json_response(self, HTTPStatus.OK, app.experiments_payload())
                    return
                if parsed.path == "/api/problem":
                    root = (query.get("root") or [""])[0].strip()
                    if root and Path(root).expanduser().resolve() != app.results_root:
                        app.results_root = Path(root).expanduser().resolve()
                        app._status_cache = None
                        app._data_cache.clear()
                        app._stage1_cache.clear()
                        app._stage2_cache.clear()
                        app._stage3_cache.clear()
                    exp_names = _split_csv((query.get("exp_names") or [""])[0])
                    parent_id = (query.get("parent_id") or [""])[0].strip()
                    _json_response(self, HTTPStatus.OK, app.problem_payload(exp_names, parent_id))
                    return
                if parsed.path == "/rollout_compare":
                    root = (query.get("root") or [""])[0].strip()
                    if root and Path(root).expanduser().resolve() != app.results_root:
                        app.results_root = Path(root).expanduser().resolve()
                        app._status_cache = None
                        app._data_cache.clear()
                        app._stage1_cache.clear()
                        app._stage2_cache.clear()
                        app._stage3_cache.clear()
                    exp_names = _split_csv((query.get("exp_names") or [""])[0])
                    parent_id = (query.get("parent_id") or [""])[0].strip()
                    rollout_id = _safe_int((query.get("rollout_id") or [""])[0])
                    body = app.render_rollout_pair_compare_html(exp_names, parent_id, rollout_id).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/mine_cases", "/api/render", "/api/compare_graph"}:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                form = _parse_post_form(self)
                root = (form.get("root") or [""])[0].strip()
                if root and Path(root).expanduser().resolve() != app.results_root:
                    app.results_root = Path(root).expanduser().resolve()
                    app._status_cache = None
                    app._data_cache.clear()
                    app._stage1_cache.clear()
                    app._stage2_cache.clear()
                    app._stage3_cache.clear()
                if parsed.path == "/api/mine_cases":
                    exp_names = _split_csv((form.get("exp_names") or [""])[0])
                    _json_response(self, HTTPStatus.OK, app.mine_cases(exp_names))
                    return
                if parsed.path == "/api/render":
                    exp_name = (form.get("exp_name") or [""])[0].strip()
                    record_id = (form.get("record_id") or [""])[0].strip()
                    html = app.render_record_html(exp_name, record_id)
                    record = app.graph_record_payload(exp_name, record_id)
                    _json_response(self, HTTPStatus.OK, {"html": html, "record": record})
                    return
                exp_names = _split_csv((form.get("exp_names") or [""])[0])
                record_id = (form.get("record_id") or [""])[0].strip()
                compare_fields = _split_csv((form.get("compare_fields") or [""])[0])
                html = app.render_compare_html(exp_names, record_id, compare_fields)
                _json_response(self, HTTPStatus.OK, {"html": html})
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for step-proof experiment comparison.")
    default_results_root = Path(__file__).resolve().parent.parent / "outputs" / "step_proofs"
    parser.add_argument("--results-root", type=Path, default=default_results_root)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--source", choices=("results", "graph"), default="results")
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()

    app = StepProofCompareApp(
        repo_root=REPO_ROOT,
        results_root=args.results_root.expanduser().resolve(),
        source=args.source,
        graph_only=bool(args.graph_only),
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"Step proof compare viewer started: http://{args.host}:{args.port}")
    print(f"results_root: {app.results_root}")
    print(f"source: {args.source}, graph_only: {args.graph_only}")
    server.serve_forever()


if __name__ == "__main__":
    main()
