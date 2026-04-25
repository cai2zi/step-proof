"""
将阶段一产出的 graph JSONL 可视化为 graph-only HTML（不展示 form/prove 状态）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

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


def _get_record_id(rec: Dict[str, Any]) -> str:
    return str(rec.get("meta", {}).get("record_id", "")).strip()


def _parse_record_ids(values: List[str]) -> List[str]:
    ids: List[str] = []
    for value in values:
        for part in value.split(","):
            rid = part.strip()
            if rid:
                ids.append(rid)
    dedup: List[str] = []
    seen = set()
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        dedup.append(rid)
    return dedup


def _select_records(
    rows: List[Dict[str, Any]],
    select_all: bool,
    random_n: int,
    record_ids: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    if not rows:
        raise SystemExit("JSONL is empty.")

    enabled_modes = int(select_all) + int(random_n > 0) + int(bool(record_ids))
    if enabled_modes != 1:
        raise SystemExit(
            "Please choose exactly one selection mode: "
            "--all OR --random-n (>0) OR --record-ids"
        )

    if select_all:
        return rows

    if record_ids:
        id_map: Dict[str, Dict[str, Any]] = {}
        for rec in rows:
            rid = _get_record_id(rec)
            if rid:
                id_map[rid] = rec
        missing = [rid for rid in record_ids if rid not in id_map]
        if missing:
            raise SystemExit(f"record_id not found: {missing}")
        return [id_map[rid] for rid in record_ids]

    if random_n > len(rows):
        raise SystemExit(f"--random-n={random_n} exceeds total rows={len(rows)}")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(rows)), random_n)
    return [rows[i] for i in idxs]


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
        "--all",
        action="store_true",
        help="Visualize all records in JSONL",
    )
    parser.add_argument(
        "--random-n",
        type=int,
        default=0,
        help="Randomly select N records for visualization",
    )
    parser.add_argument(
        "--record-ids",
        nargs="*",
        default=[],
        help="Specific record_id list (supports space or comma separation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random-n",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "calc_runs" / "HTML",
        help="Output directory, each file is named <record_id>.html",
    )
    args = parser.parse_args()

    if not args.graph_jsonl.is_file():
        raise SystemExit(f"graph JSONL not found: {args.graph_jsonl}")

    rows = _load_jsonl(args.graph_jsonl)
    record_ids = _parse_record_ids(args.record_ids)
    selected = _select_records(
        rows=rows,
        select_all=args.all,
        random_n=args.random_n,
        record_ids=record_ids,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rec in selected:
        rid = _get_record_id(rec)
        if not rid:
            raise SystemExit("Found record without meta.record_id; cannot name output file.")
        out_html = args.out_dir / f"{rid}.html"
        graph_nodes = rec.get("graph", {}).get("nodes", [])
        if not isinstance(graph_nodes, list) or not graph_nodes:
            raise SystemExit(f"record_id={rid} has empty graph.nodes")
        G, node_info = build_dag(graph_nodes)
        title, subtitle = _title_and_subtitle(rec)
        create_interactive_graph_only_visualization(
            G=G,
            node_info=node_info,
            title=title,
            subtitle=subtitle,
            filename=str(out_html),
        )


if __name__ == "__main__":
    main()
