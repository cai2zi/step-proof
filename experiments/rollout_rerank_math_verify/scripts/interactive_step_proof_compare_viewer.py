#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    ("solved_lemma.lean_code", "Solved lemma Lean"),
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

    def problem_payload(self, exp_names: List[str], parent_id: str) -> JsonDict:
        exp_names = self._normalize_exp_names(exp_names)
        parent_id = str(parent_id).strip()
        if not parent_id:
            raise ValueError("parent_id is empty")
        tables: Dict[str, JsonDict] = {}
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
                table_rows.append(
                    {
                        "rank": rank,
                        "id": rid,
                        "success_ratio": row.get("success_ratio", ""),
                        "prove_success_nodes": row.get("prove_success_nodes", ""),
                        "lean_pass_nodes": row.get("lean_pass_nodes", ""),
                        "rollout_id": rollout,
                        "math_verify_correct": _to_bool_correct(ev.get("is_correct")),
                        "selected": selected.get("rollout_id") == rollout,
                    }
                )
            tables[name] = {
                "selected": selected,
                "pass_at_1": data.pass_at_1_by_parent.get(parent_id, False),
                "pass_at_k": data.pass_at_k_by_parent.get(parent_id, False),
                "rows": table_rows,
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
    #viewer { width: 100%; height: 76vh; border: 0; background: #fff; border-radius: 8px; }
    .field-list label { display: inline-flex; gap: 4px; align-items: center; margin: 2px 8px 2px 0; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .problem-meta { grid-template-columns: 1fr 1fr; }
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
            <span id="graphStatus" class="muted"></span>
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

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
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
      const tables = data.tables || {};
      const names = data.exp_names || [];
      const meta = `<div class="problem-meta">
        <div><strong>ID</strong><br>${escapeHtml(data.parent_id)}</div>
        <div><strong>Source</strong><br>${escapeHtml(data.source)}</div>
        <div><strong>Gold</strong><br>${escapeHtml(data.gold)}</div>
        <div><strong>Experiments</strong><br>${escapeHtml(names.join(", "))}</div>
      </div>
      <h2>Question</h2>
      <div class="question">${escapeHtml(data.question)}</div>`;
      const tableHtml = names.map((name) => {
        const t = tables[name] || {};
        const selected = t.selected || {};
        return `<div class="panel">
          <div class="case-title">
            <h2>${escapeHtml(name)}</h2>
            <span class="muted">pass@1 ${t.pass_at_1 ? "yes" : "no"} / pass@4 ${t.pass_at_k ? "yes" : "no"} / selected rollout ${escapeHtml(selected.rollout_id)} ${selected.is_correct ? "correct" : "wrong"}</span>
          </div>
          <div class="table-wrap"><table>
            <thead><tr><th>rank</th><th>id</th><th>success_ratio</th><th>prove_success_nodes</th><th>lean_pass_nodes</th><th>rollout_id</th><th>math_verify</th><th>selected</th><th>graph</th></tr></thead>
            <tbody>${(t.rows || []).map((row) => `<tr>
              <td>${escapeHtml(row.rank)}</td>
              <td><a onclick="renderGraph('${escapeHtml(name)}', '${escapeHtml(row.id)}')">${escapeHtml(row.id)}</a></td>
              <td>${escapeHtml(row.success_ratio)}</td>
              <td>${escapeHtml(row.prove_success_nodes)}</td>
              <td>${escapeHtml(row.lean_pass_nodes)}</td>
              <td>${escapeHtml(row.rollout_id)}</td>
              <td>${boolMark(row.math_verify_correct)}</td>
              <td>${row.selected ? `<span class="ok">selected</span>` : ""}</td>
              <td><a onclick="renderGraph('${escapeHtml(name)}', '${escapeHtml(row.id)}')">view</a></td>
            </tr>`).join("")}</tbody>
          </table></div>
        </div>`;
      }).join("");
      document.getElementById("problem").innerHTML = meta + tableHtml;
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
                    app._stage3_cache.clear()
                    _json_response(self, HTTPStatus.OK, app.experiments_payload())
                    return
                if parsed.path == "/api/problem":
                    root = (query.get("root") or [""])[0].strip()
                    if root and Path(root).expanduser().resolve() != app.results_root:
                        app.results_root = Path(root).expanduser().resolve()
                        app._status_cache = None
                        app._data_cache.clear()
                        app._stage3_cache.clear()
                    exp_names = _split_csv((query.get("exp_names") or [""])[0])
                    parent_id = (query.get("parent_id") or [""])[0].strip()
                    _json_response(self, HTTPStatus.OK, app.problem_payload(exp_names, parent_id))
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
                    app._stage3_cache.clear()
                if parsed.path == "/api/mine_cases":
                    exp_names = _split_csv((form.get("exp_names") or [""])[0])
                    _json_response(self, HTTPStatus.OK, app.mine_cases(exp_names))
                    return
                if parsed.path == "/api/render":
                    exp_name = (form.get("exp_name") or [""])[0].strip()
                    record_id = (form.get("record_id") or [""])[0].strip()
                    html = app.render_record_html(exp_name, record_id)
                    _json_response(self, HTTPStatus.OK, {"html": html})
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
