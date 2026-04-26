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


JsonDict = Dict[str, Any]


def _remove_imports(lean_code: str) -> str:
    lines_to_remove = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]
    code_lines = lean_code.split("\n")
    filtered_lines: List[str] = []
    for line in code_lines:
        if line.strip() not in lines_to_remove:
            filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def _build_dependency_context(node: JsonDict, nodes: Dict[str, JsonDict]) -> str:
    dependencies = list(node.get("dependencies") or [])
    if not dependencies:
        return ""

    intro = (
        f"\n\n This proof step depend on previous proof steps, namely steps {dependencies}.\n"
        "Please make use use of their formal lean4 code, which contains relevant lean4 "
        "hypothesis and type declarations you may use:"
    )
    parts: List[str] = []
    for dep_id in dependencies:
        dep = nodes.get(dep_id)
        if dep is None:
            continue
        formalization = dep.get("formalization") or {}
        if formalization.get("lean_code") and formalization.get("lean_pass"):
            parts.append("\n")
            parts.append(_remove_imports(formalization["lean_code"]))
        else:
            parts.append(
                "\nDependency step "
                f"{dep_id} is provided in natural language: \"{dep.get('statement', '')}\". "
                "Please formalize it as part of your current lemma's hypotheses."
            )

    if not parts:
        return ""

    footer = (
        "\nFocus on the original formalization task I gave you and use the previous Lean codes, "
        "extra context, type declarations, variables domains, etc. You can assume the "
        "information is correct. Make use of it!"
    )
    return (intro + "\n".join(parts) + footer).strip()


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
    if source == "results":
        nodes = rec.get("results", {}).get("nodes", [])
        if (not isinstance(nodes, list) or not nodes) and rec.get("graph", {}).get("nodes"):
            nodes = rec.get("graph", {}).get("nodes", [])
    elif source == "graph":
        nodes = rec.get("graph", {}).get("nodes", [])
        if (not isinstance(nodes, list) or not nodes) and rec.get("results", {}).get("nodes"):
            nodes = rec.get("results", {}).get("nodes", [])
    else:
        raise ValueError(f"invalid source: {source}")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"Selected record has empty {source}.nodes and no fallback node list")

    nodes_dict = {n["id"]: n for n in nodes if "id" in n}
    for n in nodes:
        if "formalization" not in n or not isinstance(n["formalization"], dict):
            n["formalization"] = {}
        if not n["formalization"].get("dependency_context_block"):
            n["formalization"]["dependency_context_block"] = _build_dependency_context(n, nodes_dict)
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
        for rec in _load_jsonl(stage3_jsonl):
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
      <div class="row" id="expOptions"></div>
      <div class="row" style="margin-top:8px;">
        <label for="recordId"><strong>record_id:</strong></label>
        <input id="recordId" type="text" placeholder="输入 record_id，例如 12345" />
        <button id="btnRender">确定并展示</button>
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
        const id = `exp_${idx}`;
        const label = document.createElement('label');
        label.style.marginRight = '12px';
        label.innerHTML = `<input type="radio" name="exp_name" value="${name}" ${idx===0 ? 'checked' : ''}/> ${name}`;
        expOptions.appendChild(label);
      });
    }

    function selectedExpName() {
      const selected = document.querySelector('input[name="exp_name"]:checked');
      return selected ? selected.value : '';
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
    document.getElementById('btnBack').addEventListener('click', goBack);
    document.getElementById('recordId').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') renderRecord();
    });

    loadExperiments().catch((err) => {
      document.getElementById('error').textContent = err.message || String(err);
    });
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
            if parsed.path != "/api/render":
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
            record_id = (form.get("record_id") or [""])[0].strip()

            try:
                html = app.render_record_html(exp_name=exp_name, record_id=record_id)
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
