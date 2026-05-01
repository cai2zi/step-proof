#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from proofflow.vis import (
    build_dag,
    create_interactive_graph_only_visualization,
    create_interactive_visualization,
)
from proofflow.graph_mode import FDG_GRAPH_MODE, ensure_single_graph_mode, extract_record_items


JsonDict = Dict[str, Any]
COMPARE_FIELD_OPTIONS = [
    ("natural_language", "Natural Language"),
    ("statement", "Statement"),
    ("text", "Fact Text"),
    ("proof_obligation.informal_statement_content", "Proof Obligation"),
    ("formalization.lean_code", "Formalization.lean_code"),
    ("formalization.dependency_context_block", "Formalization.dependency_context_block"),
    ("solved_lemma.lean_code", "Solved_lemma.lean_code"),
]


def _load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_nodes(rec: JsonDict, source: str) -> List[JsonDict]:
    nodes, graph_mode = extract_record_items(rec, source)
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"Selected record has empty {source} items and no fallback list")

    if graph_mode != FDG_GRAPH_MODE:
        raise ValueError(f"Only FDG records are supported, got graph_mode={graph_mode!r}")
    return nodes


def _build_proof_str(rec: JsonDict) -> str:
    inp = rec.get("input", {})
    problem = str(inp.get("problem", "")).strip()
    raw_cot = str(inp.get("raw_cot", "")).strip()
    answer = str(inp.get("answer", "")).strip()

    parts: List[str] = []
    if problem:
        parts.append("Problem:")
        parts.append(problem)
    if raw_cot:
        parts.append("")
        parts.append("Raw CoT:")
        parts.append(raw_cot)
    if answer and not raw_cot:
        parts.append("")
        parts.append("Answer:")
        parts.append(answer)
    return "\n".join(parts)


def _get_nested_value(payload: JsonDict, field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
        if current is None:
            return ""
    return current


class ViewerApp:
    def __init__(self, repo_root: Path, results_root: Path, source: str, graph_only: bool) -> None:
        self.repo_root = repo_root
        self.results_root = results_root
        self.source = source
        self.graph_only = graph_only
        self.cache_dir = self.repo_root / ".tmp_stage3_viewer_html"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()
        self._record_cache: Dict[str, Dict[str, JsonDict]] = {}

    def list_experiments(self) -> List[str]:
        if not self.results_root.is_dir():
            return []
        names = [
            p.name
            for p in self.results_root.iterdir()
            if p.is_dir() and (p / "result_stage3" / "stage3_results.jsonl").is_file()
        ]
        names.sort()
        return names

    def _load_exp_records(self, exp_name: str) -> Dict[str, JsonDict]:
        if exp_name in self._record_cache:
            return self._record_cache[exp_name]

        stage3_jsonl = self.results_root / exp_name / "result_stage3" / "stage3_results.jsonl"
        if not stage3_jsonl.is_file():
            raise FileNotFoundError(f"missing stage3 file: {stage3_jsonl}")

        id_map: Dict[str, JsonDict] = {}
        rows = _load_jsonl(stage3_jsonl)
        ensure_single_graph_mode(rows, source_name=str(stage3_jsonl))
        for rec in rows:
            rid = str(rec.get("meta", {}).get("record_id", "")).strip()
            if rid:
                id_map[rid] = rec

        self._record_cache[exp_name] = id_map
        return id_map

    def render_record_html(self, exp_name: str, record_id: str) -> str:
        record_id = str(record_id).strip()
        if not record_id:
            raise ValueError("record_id is empty")

        id_map = self._load_exp_records(exp_name)
        rec = id_map.get(record_id)
        if rec is None:
            raise KeyError(f"record_id not found in {exp_name}: {record_id}")

        nodes = _extract_nodes(rec, self.source)
        G, node_info = build_dag(nodes)
        out_path = self.cache_dir / f"{exp_name}__{record_id}.html"
        with self._cache_lock:
            if self.graph_only:
                title = f"Stage3 DAG (record_id={record_id}, source={self.source})"
                subtitle = f"exp_name: {exp_name}"
                create_interactive_graph_only_visualization(
                    G=G,
                    node_info=node_info,
                    title=title,
                    subtitle=subtitle,
                    filename=str(out_path),
                )
            else:
                proof_str = _build_proof_str(rec)
                create_interactive_visualization(
                    G=G,
                    node_info=node_info,
                    proof_str=proof_str,
                    filename=str(out_path),
                )
        return out_path.read_text(encoding="utf-8")

    def render_compare_html(
        self,
        exp_names: List[str],
        record_id: str,
        compare_fields: List[str],
    ) -> str:
        exp_names = [str(name).strip() for name in exp_names if str(name).strip()]
        compare_fields = [str(name).strip() for name in compare_fields if str(name).strip()]
        record_id = str(record_id).strip()
        if not exp_names:
            raise ValueError("exp_names is empty")
        if not record_id:
            raise ValueError("record_id is empty")
        if not compare_fields:
            raise ValueError("compare_fields is empty")

        base_exp_name = exp_names[0]
        base_rec = self._load_exp_records(base_exp_name).get(record_id)
        if base_rec is None:
            raise KeyError(f"record_id not found in {base_exp_name}: {record_id}")

        nodes = _extract_nodes(base_rec, self.source)
        G, node_info = build_dag(nodes)
        compare_payload: Dict[str, Dict[str, Dict[str, Any]]] = {
            node_id: {} for node_id in node_info.keys()
        }

        for exp_name in exp_names:
            rec = self._load_exp_records(exp_name).get(record_id)
            if rec is None:
                raise KeyError(f"record_id not found in {exp_name}: {record_id}")
            exp_nodes = _extract_nodes(rec, self.source)
            exp_node_map = {str(node.get("id", "")).strip(): node for node in exp_nodes}
            for node_id in compare_payload.keys():
                node_payload = exp_node_map.get(node_id, {})
                compare_payload[node_id][exp_name] = {
                    field_name: _get_nested_value(node_payload, field_name)
                    for field_name in compare_fields
                }

        out_path = self.cache_dir / f"compare__{'__'.join(exp_names)}__{record_id}.html"
        with self._cache_lock:
            proof_str = _build_proof_str(base_rec)
            create_interactive_visualization(
                G=G,
                node_info=node_info,
                proof_str=proof_str,
                filename=str(out_path),
                compare_payload=compare_payload,
                compare_fields=compare_fields,
                compare_experiments=exp_names,
            )
        return out_path.read_text(encoding="utf-8")


def _html_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stage3 可视化交互页面</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f7fb; color: #222; }
    .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
    .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    input[type="text"] { padding: 6px 8px; min-width: 280px; }
    button { padding: 7px 12px; cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: 0.5; }
    .muted { color: #666; font-size: 13px; }
    #viewerWrap { background: #fff; border: 1px solid #ddd; border-radius: 8px; min-height: 600px; }
    #viewer { width: 100%; height: 76vh; border: 0; border-radius: 8px; }
    .error { color: #c62828; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="row"><strong>Experiments:</strong></div>
      <div class="row" id="expOptions"></div>
      <div class="row" style="margin-top:8px;"><strong>Compare Fields:</strong></div>
      <div class="row" id="fieldOptions"></div>
      <div class="row" style="margin-top:6px;">
        <span class="muted">左侧图使用第一个选中的实验，右侧可聚合多实验信息做横向对比。</span>
      </div>
      <div class="row" style="margin-top:8px;">
        <label for="recordId"><strong>record_id:</strong></label>
        <input id="recordId" type="text" placeholder="输入 record_id，例如 12345" />
        <button id="btnRender">确定并展示</button>
        <button id="btnCompare">对比展示</button>
        <button id="btnBack" disabled>返回上一条</button>
        <span id="status" class="muted"></span>
      </div>
      <div id="error" class="error"></div>
    </div>
    <div id="viewerWrap">
      <iframe id="viewer" title="stage3-graph-viewer"></iframe>
    </div>
  </div>

  <script>
    const renderHistory = [];
    const compareFieldOptions = """ + json.dumps(
        [{"value": value, "label": label} for value, label in COMPARE_FIELD_OPTIONS],
        ensure_ascii=False,
    ) + """;

    async function getJSON(url, options) {
      const resp = await fetch(url, options);
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || 'Request failed');
      }
      return data;
    }

    async function loadExperiments() {
      const data = await getJSON('/api/experiments');
      const expOptions = document.getElementById('expOptions');
      expOptions.innerHTML = '';
      if (!data.experiments.length) {
        expOptions.innerHTML = '<span class="error">未发现可用实验（需要 results/*/result_stage3/stage3_results.jsonl）</span>';
        return;
      }
      data.experiments.forEach((name, idx) => {
        const label = document.createElement('label');
        label.style.marginRight = '12px';
        label.innerHTML = `<input type="checkbox" name="exp_name" value="${name}" ${idx===0 ? 'checked' : ''}/> ${name}`;
        expOptions.appendChild(label);
      });
    }

    function loadFieldOptions() {
      const fieldOptions = document.getElementById('fieldOptions');
      fieldOptions.innerHTML = '';
      compareFieldOptions.forEach((item, idx) => {
        const label = document.createElement('label');
        label.style.marginRight = '12px';
        label.innerHTML = `<input type="checkbox" name="compare_field" value="${item.value}" ${idx < 2 ? 'checked' : ''}/> ${item.label}`;
        fieldOptions.appendChild(label);
      });
    }

    function selectedExpNames() {
      return Array.from(document.querySelectorAll('input[name="exp_name"]:checked')).map((el) => el.value);
    }

    function selectedExpName() {
      const selected = selectedExpNames();
      return selected.length ? selected[0] : '';
    }

    function selectedCompareFields() {
      return Array.from(document.querySelectorAll('input[name="compare_field"]:checked')).map((el) => el.value);
    }

    function updateBackButton() {
      document.getElementById('btnBack').disabled = renderHistory.length === 0;
    }

    async function renderRecordByTarget(exp_name, record_id, pushCurrent) {
      const errorEl = document.getElementById('error');
      const statusEl = document.getElementById('status');
      errorEl.textContent = '';
      if (!exp_name) {
        errorEl.textContent = '请先选择实验名';
        return;
      }
      if (!record_id) {
        errorEl.textContent = '请输入 record_id';
        return;
      }

      const currentExp = selectedExpName();
      const currentRecord = document.getElementById('recordId').value.trim();
      if (pushCurrent && currentExp && currentRecord) {
        renderHistory.push({ exp_name: currentExp, record_id: currentRecord });
      }

      statusEl.textContent = '渲染中...';
      try {
        const data = await getJSON('/api/render', {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: `exp_name=${encodeURIComponent(exp_name)}&record_id=${encodeURIComponent(record_id)}`
        });
        const iframe = document.getElementById('viewer');
        iframe.srcdoc = data.html;
        const expRadio = document.querySelector(`input[name="exp_name"][value="${exp_name}"]`);
        if (expRadio) expRadio.checked = true;
        document.getElementById('recordId').value = record_id;
        statusEl.textContent = `已展示: ${exp_name} / ${record_id}`;
        updateBackButton();
      } catch (err) {
        if (pushCurrent && currentExp && currentRecord) {
          renderHistory.pop();
        }
        errorEl.textContent = err.message || String(err);
        statusEl.textContent = '';
        updateBackButton();
      }
    }

    async function renderRecord() {
      const exp_name = selectedExpName();
      const record_id = document.getElementById('recordId').value.trim();
      await renderRecordByTarget(exp_name, record_id, true);
    }

    async function renderCompare() {
      const expNames = selectedExpNames();
      const compareFields = selectedCompareFields();
      const record_id = document.getElementById('recordId').value.trim();
      const errorEl = document.getElementById('error');
      const statusEl = document.getElementById('status');
      errorEl.textContent = '';
      if (!expNames.length) {
        errorEl.textContent = '请至少选择一个实验名';
        return;
      }
      if (!compareFields.length) {
        errorEl.textContent = '请至少选择一个对比字段';
        return;
      }
      if (!record_id) {
        errorEl.textContent = '请输入 record_id';
        return;
      }

      const currentExp = selectedExpName();
      const currentRecord = document.getElementById('recordId').value.trim();
      if (currentExp && currentRecord) {
        renderHistory.push({ exp_name: currentExp, record_id: currentRecord });
      }

      statusEl.textContent = '对比渲染中...';
      try {
        const data = await getJSON('/api/compare', {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: `exp_names=${encodeURIComponent(expNames.join(','))}&record_id=${encodeURIComponent(record_id)}&compare_fields=${encodeURIComponent(compareFields.join(','))}`
        });
        document.getElementById('viewer').srcdoc = data.html;
        document.getElementById('recordId').value = record_id;
        statusEl.textContent = `已对比展示: ${expNames.join(', ')} / ${record_id}`;
        updateBackButton();
      } catch (err) {
        if (currentExp && currentRecord) {
          renderHistory.pop();
        }
        errorEl.textContent = err.message || String(err);
        statusEl.textContent = '';
        updateBackButton();
      }
    }

    async function goBack() {
      if (!renderHistory.length) return;
      const previous = renderHistory.pop();
      updateBackButton();
      await renderRecordByTarget(previous.exp_name, previous.record_id, false);
    }

    window.addEventListener('message', (event) => {
      const data = event && event.data;
      if (!data || data.type !== 'viewer-back') return;
      goBack();
    });

    document.getElementById('btnRender').addEventListener('click', renderRecord);
    document.getElementById('btnCompare').addEventListener('click', renderCompare);
    document.getElementById('btnBack').addEventListener('click', goBack);
    document.getElementById('recordId').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') renderRecord();
    });

    loadExperiments().catch((err) => {
      document.getElementById('error').textContent = err.message || String(err);
    });
    loadFieldOptions();
    updateBackButton();
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


def create_handler(app: ViewerApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _html_page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/experiments":
                _json_response(self, HTTPStatus.OK, {"experiments": app.list_experiments()})
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/render", "/api/compare"}:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return
            raw = self.rfile.read(content_length).decode("utf-8")
            form = parse_qs(raw, keep_blank_values=True)
            exp_name = (form.get("exp_name") or [""])[0].strip()
            exp_names = [
                item.strip()
                for item in ((form.get("exp_names") or [""])[0].split(","))
                if item.strip()
            ]
            compare_fields = [
                item.strip()
                for item in ((form.get("compare_fields") or [""])[0].split(","))
                if item.strip()
            ]
            record_id = (form.get("record_id") or [""])[0].strip()

            try:
                if parsed.path == "/api/render":
                    html = app.render_record_html(exp_name=exp_name, record_id=record_id)
                else:
                    html = app.render_compare_html(
                        exp_names=exp_names,
                        record_id=record_id,
                        compare_fields=compare_fields,
                    )
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return

            _json_response(self, HTTPStatus.OK, {"html": html})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式 Stage3 结果可视化页面")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="results 根目录，默认 <repo>/results",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="监听端口，默认 8765")
    parser.add_argument(
        "--source",
        choices=("results", "graph"),
        default="results",
        help="使用 results.nodes 或 graph.nodes",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="使用 graph-only 视图",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    app = ViewerApp(
        repo_root=repo_root,
        results_root=args.results_root.resolve(),
        source=args.source,
        graph_only=bool(args.graph_only),
    )

    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"Stage3 viewer started: http://{args.host}:{args.port}")
    print(f"results_root: {app.results_root}")
    print(f"source: {args.source}, graph_only: {args.graph_only}")
    server.serve_forever()


if __name__ == "__main__":
    main()
