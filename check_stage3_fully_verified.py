from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from proofflow.graph_mode import FDG_GRAPH_MODE, ensure_single_graph_mode


JsonDict = Dict[str, Any]


def _load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _node_fully_verified(node: JsonDict) -> bool:
    if "fact_id" in node:
        if node.get("parent_fact_ids"):
            return bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        return bool((node.get("formalization") or {}).get("lean_pass", False))
    role = str(node.get("role", "")).strip()
    if role in {"condition", "context"}:
        return bool((node.get("formalization") or {}).get("lean_pass", False))
    return bool((node.get("solved_lemma") or {}).get("lean_verify", False))


def _record_items(rec: JsonDict, graph_mode: str) -> List[JsonDict]:
    key = "facts" if graph_mode == FDG_GRAPH_MODE else "nodes"
    items = rec.get("results", {}).get(key, [])
    return items if isinstance(items, list) else []


def _prove_required_items(items: List[JsonDict], graph_mode: str) -> List[JsonDict]:
    if graph_mode == FDG_GRAPH_MODE:
        return [item for item in items if bool(item.get("parent_fact_ids"))]
    return [item for item in items if int(item.get("needs_verification", 1)) != 0]


def _form_required_items(items: List[JsonDict], graph_mode: str) -> List[JsonDict]:
    if graph_mode == FDG_GRAPH_MODE:
        return [item for item in items if bool(item.get("parent_fact_ids"))]
    return items


def _record_all_nodes_prove_verified(rec: JsonDict, graph_mode: str) -> bool:
    nodes = _record_items(rec, graph_mode)
    if not nodes:
        return False
    required_nodes = _prove_required_items(nodes, graph_mode)
    if not required_nodes:
        return False
    return all(
        bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        for node in required_nodes
    )


def _bucket_label_by_percent(percent: float) -> str:
    clipped = max(0.0, min(100.0, percent))
    if clipped >= 100.0:
        return "100%"
    left = int(clipped // 10) * 10
    right = left + 10
    return f"{left}-{right}%"


def _empty_histogram() -> Dict[str, int]:
    hist = {f"{i}-{i + 10}%": 0 for i in range(0, 100, 10)}
    hist["100%"] = 0
    return hist


def _bucket_order() -> List[str]:
    return [f"{i}-{i + 10}%" for i in range(0, 100, 10)] + ["100%"]


def _print_histogram(title: str, hist: Dict[str, int]) -> None:
    print(title)
    for key in _bucket_order():
        if key == "100%":
            print(f"  100%: {hist['100%']}")
            continue
        print(f"  {key}: {hist[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 verification statistics by record and by node.",
    )
    parser.add_argument(
        "--stage3-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "result_stage3" / "stage3_results.jsonl",
        help="Path to stage3 results JSONL",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print first 5 record_ids for each prove-verify ratio bucket.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "result_stage3" / "stage3_verify_stats.json",
        help="Path to save statistics JSON output.",
    )
    parser.add_argument(
        "--top-n-per-bucket",
        type=int,
        default=5,
        help="Number of record_ids to keep for each prove-verify ratio bucket.",
    )
    args = parser.parse_args()

    if not args.stage3_jsonl.is_file():
        raise SystemExit(f"stage3 JSONL not found: {args.stage3_jsonl}")

    rows = _load_jsonl(args.stage3_jsonl)
    graph_mode = ensure_single_graph_mode(rows, source_name=str(args.stage3_jsonl))
    total_records = len(rows)
    valid_records = 0

    passed_ids: List[str] = []
    prove_ratio_hist = _empty_histogram()
    form_ratio_hist = _empty_histogram()
    prove_bucket_ids: Dict[str, List[str]] = {k: [] for k in _bucket_order()}
    final_answer_wrong_records = 0

    total_nodes = 0
    total_prove_verified_nodes = 0
    total_prove_required_nodes = 0
    total_form_verified_nodes = 0
    total_form_required_nodes = 0

    for rec in rows:
        rid = str(rec.get("meta", {}).get("record_id", "")).strip()
        nodes = _record_items(rec, graph_mode)
        if not isinstance(nodes, list) or not nodes:
            continue

        valid_records += 1
        node_count = len(nodes)
        total_nodes += node_count

        prove_required_nodes = _prove_required_items(nodes, graph_mode)
        prove_required_count = len(prove_required_nodes)
        prove_verified = sum(
            1
            for node in prove_required_nodes
            if bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        )
        form_required_nodes = _form_required_items(nodes, graph_mode)
        form_required_count = len(form_required_nodes)
        form_verified = sum(
            1
            for node in form_required_nodes
            if bool((node.get("formalization") or {}).get("lean_pass", False))
        )
        total_prove_required_nodes += prove_required_count
        total_prove_verified_nodes += prove_verified
        total_form_verified_nodes += form_verified
        total_form_required_nodes += form_required_count

        prove_percent = (
            (prove_verified / prove_required_count) * 100.0
            if prove_required_count
            else 0.0
        )
        form_percent = (form_verified / form_required_count) * 100.0 if form_required_count else 0.0
        prove_bucket = _bucket_label_by_percent(prove_percent)
        form_bucket = _bucket_label_by_percent(form_percent)
        prove_ratio_hist[prove_bucket] += 1
        form_ratio_hist[form_bucket] += 1
        prove_bucket_ids[prove_bucket].append(rid)

        if _record_all_nodes_prove_verified(rec, graph_mode):
            passed_ids.append(rid)

        if graph_mode == FDG_GRAPH_MODE:
            final_nodes_required = [node for node in nodes if bool(node.get("is_final_answer"))]
        else:
            final_nodes_required = [
                node
                for node in nodes
                if str(node.get("role", "")).strip() == "final"
                and int(node.get("needs_verification", 1)) != 0
            ]
        final_wrong = any(
            not bool((node.get("solved_lemma") or {}).get("lean_verify", False))
            for node in final_nodes_required
        )
        if final_wrong:
            final_answer_wrong_records += 1

    passed = len(passed_ids)
    passed_ratio = (passed / valid_records * 100.0) if valid_records else 0.0
    final_answer_wrong_ratio = (
        (final_answer_wrong_records / valid_records) * 100.0 if valid_records else 0.0
    )
    prove_global_ratio = (
        (total_prove_verified_nodes / total_prove_required_nodes) * 100.0
        if total_prove_required_nodes
        else 0.0
    )
    form_global_ratio = (
        (total_form_verified_nodes / total_form_required_nodes) * 100.0
        if total_form_required_nodes
        else 0.0
    )

    print(f"total_records_in_jsonl: {total_records}")
    print(f"valid_records_with_nodes: {valid_records}")
    print()
    print(f"all_nodes_prove_verified_records: {passed}")
    print(f"all_nodes_prove_verified_records_ratio: {passed_ratio:.2f}%")
    print()
    _print_histogram(
        "prove_verify_ratio_distribution_by_record (10% step):",
        prove_ratio_hist,
    )
    print()
    _print_histogram(
        "form_verify_ratio_distribution_by_record (10% step):",
        form_ratio_hist,
    )
    print()
    print(f"final_answer_wrong_records: {final_answer_wrong_records}")
    print(f"final_answer_wrong_records_ratio: {final_answer_wrong_ratio:.2f}%")
    print()
    print(f"global_nodes_total: {total_nodes}")
    print(f"global_prove_required_nodes_total: {total_prove_required_nodes}")
    print(
        "global_prove_verified_nodes: "
        f"{total_prove_verified_nodes} ({prove_global_ratio:.2f}%)"
    )
    print(
        "global_form_verified_nodes: "
        f"{total_form_verified_nodes} ({form_global_ratio:.2f}%)"
    )

    stats_payload: JsonDict = {
        "graph_mode": graph_mode,
        "total_records_in_jsonl": total_records,
        "valid_records_with_nodes": valid_records,
        "all_nodes_prove_verified_records": passed,
        "all_nodes_prove_verified_records_ratio": round(passed_ratio, 6),
        "final_answer_wrong_records": final_answer_wrong_records,
        "final_answer_wrong_records_ratio": round(final_answer_wrong_ratio, 6),
        "prove_verify_ratio_distribution_by_record": prove_ratio_hist,
        "form_verify_ratio_distribution_by_record": form_ratio_hist,
        "global_nodes_total": total_nodes,
        "global_prove_required_nodes_total": total_prove_required_nodes,
        "global_form_required_nodes_total": total_form_required_nodes,
        "global_prove_verified_nodes": total_prove_verified_nodes,
        "global_prove_verified_nodes_ratio": round(prove_global_ratio, 6),
        "global_form_verified_nodes": total_form_verified_nodes,
        "global_form_verified_nodes_ratio": round(form_global_ratio, 6),
        "prove_verify_ratio_distribution_top_ids": {
            key: ids[: max(args.top_n_per_bucket, 0)]
            for key, ids in prove_bucket_ids.items()
        },
        "prove_verify_ratio_distribution_top5_ids": {
            key: ids[:5] for key, ids in prove_bucket_ids.items()
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"stats_json_saved_to: {args.out_json}")

    if args.show_ids:
        print(f"prove_verify_ratio_distribution_by_record top{args.top_n_per_bucket} ids:")
        for key in _bucket_order():
            top_ids = prove_bucket_ids[key][: max(args.top_n_per_bucket, 0)]
            joined = ", ".join(top_ids) if top_ids else "(none)"
            print(f"  {key}: {joined}")


if __name__ == "__main__":
    main()
