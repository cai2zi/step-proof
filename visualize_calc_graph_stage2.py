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
from proofflow.stage2_common import build_dependency_context


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


def _parse_bucket_labels(values: List[str]) -> List[str]:
    return _parse_record_ids(values)


def _load_ids_from_stage3_stats(stats_json: Path, buckets: List[str]) -> List[str]:
    if not stats_json.is_file():
        raise SystemExit(f"stage3 stats JSON not found: {stats_json}")
    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    bucket_to_ids = payload.get("prove_verify_ratio_distribution_top_ids")
    if not isinstance(bucket_to_ids, dict):
        bucket_to_ids = payload.get("prove_verify_ratio_distribution_top5_ids")
    if not isinstance(bucket_to_ids, dict):
        raise SystemExit(
            "Invalid stage3 stats JSON: missing prove_verify_ratio_distribution_top_ids"
        )

    missing = [bucket for bucket in buckets if bucket not in bucket_to_ids]
    if missing:
        raise SystemExit(f"bucket not found in stage3 stats JSON: {missing}")

    selected: List[str] = []
    for bucket in buckets:
        ids = bucket_to_ids.get(bucket)
        if not isinstance(ids, list):
            raise SystemExit(f"Invalid bucket entry for {bucket}: expected id list")
        for rid in ids:
            rid_str = str(rid).strip()
            if rid_str:
                selected.append(rid_str)
    return _parse_record_ids(selected)


def _select_records(
    rows: List[Dict[str, Any]],
    random_n: int,
    record_ids: List[str],
    stats_bucket_ids: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    if not rows:
        raise SystemExit("JSONL is empty.")

    mode_count = int(random_n > 0) + int(bool(record_ids)) + int(bool(stats_bucket_ids))
    if mode_count > 1:
        raise SystemExit(
            "Only one selection mode is allowed: "
            "--random-n OR --record-ids OR --prove-ratio-buckets"
        )

    selected_ids = record_ids if record_ids else stats_bucket_ids
    if selected_ids:
        id_map: Dict[str, Dict[str, Any]] = {}
        for rec in rows:
            rid = _get_record_id(rec)
            if rid:
                id_map[rid] = rec
        missing = [rid for rid in selected_ids if rid not in id_map]
        if missing:
            raise SystemExit(f"record_id not found: {missing}")
        return [id_map[rid] for rid in selected_ids]

    if random_n <= 0:
        raise SystemExit(
            "Please provide --random-n (>0), --record-ids, or --prove-ratio-buckets"
        )
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


def _build_proof_str(rec: Dict[str, Any]) -> str:
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


def _extract_nodes(rec: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    if source == "results":
        nodes = rec.get("results", {}).get("nodes", [])
        if (not isinstance(nodes, list) or not nodes) and rec.get("graph", {}).get("nodes"):
            nodes = rec.get("graph", {}).get("nodes", [])
    elif source == "graph":
        nodes = rec.get("graph", {}).get("nodes", [])
        if (not isinstance(nodes, list) or not nodes) and rec.get("results", {}).get("nodes"):
            nodes = rec.get("results", {}).get("nodes", [])
    else:
        raise SystemExit(f"invalid --source: {source} (expected: results|graph)")
    if not isinstance(nodes, list) or not nodes:
        raise SystemExit(f"Selected record has empty {source}.nodes and no fallback node list")

    nodes_dict = {n["id"]: n for n in nodes}
    for n in nodes:
        if "formalization" not in n or not isinstance(n["formalization"], dict):
            n["formalization"] = {}
        if not n["formalization"].get("dependency_context_block"):
            n["formalization"]["dependency_context_block"] = build_dependency_context(n, nodes_dict)

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
        "--stage3-stats-json",
        type=Path,
        default=Path(__file__).resolve().parent / "result_stage3" / "stage3_verify_stats.json",
        help="Path to stage3_verify_stats.json",
    )
    parser.add_argument(
        "--prove-ratio-buckets",
        nargs="*",
        default=[],
        help=(
            "Read ids from prove_verify_ratio_distribution_top5_ids by bucket labels "
            "(supports space or comma separation, e.g. 90-100% or 100%)"
        ),
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
    bucket_labels = _parse_bucket_labels(args.prove_ratio_buckets)
    stats_bucket_ids = []
    if bucket_labels:
        stats_bucket_ids = _load_ids_from_stage3_stats(args.stage3_stats_json, bucket_labels)
        if not stats_bucket_ids:
            raise SystemExit(
                "No record ids found for the selected --prove-ratio-buckets in stage3 stats JSON."
            )
    selected = _select_records(
        rows=rows,
        random_n=args.random_n,
        record_ids=record_ids,
        stats_bucket_ids=stats_bucket_ids,
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

        # 左侧信息面板展示问题 + 原始推理（raw_cot）。
        proof_str = _build_proof_str(rec)
        create_interactive_visualization(
            G=G,
            node_info=node_info,
            proof_str=proof_str,
            filename=str(out_html),
        )


if __name__ == "__main__":
    main()
