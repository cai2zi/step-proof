"""
将阶段一产出的 graph JSONL 可视化为 graph-only HTML（不展示 form/prove 状态）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from proofflow.vis import build_dag, create_interactive_graph_only_visualization


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_record(
    rows: List[Dict[str, Any]],
    index: int,
    record_id: str,
) -> Dict[str, Any]:
    if record_id:
        for rec in rows:
            rid = str(rec.get("meta", {}).get("record_id", ""))
            if rid == record_id:
                return rec
        raise SystemExit(f"record_id not found: {record_id}")

    if not rows:
        raise SystemExit("JSONL is empty.")
    if index < 0 or index >= len(rows):
        raise SystemExit(f"--index out of range: {index} (0..{len(rows)-1})")
    return rows[index]


def _title_and_subtitle(rec: Dict[str, Any]) -> tuple[str, str]:
    meta = rec.get("meta", {})
    inp = rec.get("input", {})
    rid = meta.get("record_id", "")
    src = meta.get("source_file", "")
    problem = str(inp.get("problem", ""))
    p_preview = problem if len(problem) <= 220 else (problem[:220] + " ...")
    title = f"Graph-Only DAG (record_id={rid})"
    subtitle = f"source: {src}\nproblem: {p_preview}"
    return title, subtitle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize stage1 graph JSONL as graph-only HTML.",
    )
    parser.add_argument(
        "--graph-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "calc_runs" / "graphs.jsonl",
        help="Path to graph JSONL from stage1",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Record index in JSONL (ignored when --record-id is set)",
    )
    parser.add_argument(
        "--record-id",
        default="",
        help="Select record by meta.record_id",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path(__file__).resolve().parent / "calc_runs" / "graph_only_dag.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    if not args.graph_jsonl.is_file():
        raise SystemExit(f"graph JSONL not found: {args.graph_jsonl}")

    rows = _load_jsonl(args.graph_jsonl)
    rec = _select_record(rows, args.index, args.record_id)

    graph_nodes = rec.get("graph", {}).get("nodes", [])
    if not isinstance(graph_nodes, list) or not graph_nodes:
        raise SystemExit("Selected record has empty graph.nodes")

    G, node_info = build_dag(graph_nodes)
    title, subtitle = _title_and_subtitle(rec)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    create_interactive_graph_only_visualization(
        G=G,
        node_info=node_info,
        title=title,
        subtitle=subtitle,
        filename=str(args.out_html),
    )


if __name__ == "__main__":
    main()
