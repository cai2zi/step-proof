"""
将阶段二产出的 stage2 JSONL 可视化为 HTML。

- 默认展示 results.nodes（包含 form/prove 状态）。
- 可切换到 graph.nodes（仅图结构字段）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from proofflow.vis import (
    build_dag,
    create_interactive_graph_only_visualization,
    create_interactive_visualization,
)


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
    random_n: int,
    record_ids: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    if not rows:
        raise SystemExit("JSONL is empty.")

    if random_n > 0 and record_ids:
        raise SystemExit("Only one selection mode is allowed: --random-n OR --record-ids")

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

    if random_n <= 0:
        raise SystemExit("Please provide --random-n (>0) or --record-ids")
    if random_n > len(rows):
        raise SystemExit(f"--random-n={random_n} exceeds total rows={len(rows)}")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(rows)), random_n)
    return [rows[i] for i in idxs]


def _title_and_subtitle(rec: Dict[str, Any], source: str) -> tuple[str, str]:
    meta = rec.get("meta", {})
    inp = rec.get("input", {})
    exe = rec.get("execution", {})
    rid = meta.get("record_id", "")
    src = meta.get("source_file", "")
    status = exe.get("record_status", "unknown")
    problem = str(inp.get("problem", ""))
    p_preview = problem if len(problem) <= 220 else (problem[:220] + " ...")
    title = f"Stage2 DAG (record_id={rid}, source={source})"
    subtitle = f"source_file: {src}\nrecord_status: {status}\nproblem: {p_preview}"
    return title, subtitle


def _extract_nodes(rec: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    if source == "results":
        nodes = rec.get("results", {}).get("nodes", [])
    elif source == "graph":
        nodes = rec.get("graph", {}).get("nodes", [])
    else:
        raise SystemExit(f"invalid --source: {source} (expected: results|graph)")
    if not isinstance(nodes, list) or not nodes:
        raise SystemExit(f"Selected record has empty {source}.nodes")
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize stage2 JSONL as interactive DAG HTML.",
    )
    parser.add_argument(
        "--stage2-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "calc_runs" / "stage2_results.jsonl",
        help="Path to stage2 results JSONL",
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
        "--source",
        choices=("results", "graph"),
        default="results",
        help="Which node list to visualize: results.nodes or graph.nodes",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Use graph-only renderer (hide verification-status coloring)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/czx/step-proof/calc_runs/HTML"),
        help="Output directory, each file is named <record_id>.html",
    )
    args = parser.parse_args()

    if not args.stage2_jsonl.is_file():
        raise SystemExit(f"stage2 JSONL not found: {args.stage2_jsonl}")

    rows = _load_jsonl(args.stage2_jsonl)
    record_ids = _parse_record_ids(args.record_ids)
    selected = _select_records(
        rows=rows,
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
        nodes = _extract_nodes(rec, args.source)
        G, node_info = build_dag(nodes)
        title, subtitle = _title_and_subtitle(rec, args.source)

        if args.graph_only:
            create_interactive_graph_only_visualization(
                G=G,
                node_info=node_info,
                title=title,
                subtitle=subtitle,
                filename=str(out_html),
            )
            continue

        # results 模式默认展示 form/prove 状态（边框颜色），graph 模式同样可渲染。
        proof_str = str(rec.get("input", {}).get("problem", ""))
        create_interactive_visualization(
            G=G,
            node_info=node_info,
            proof_str=proof_str,
            filename=str(out_html),
        )


if __name__ == "__main__":
    main()
