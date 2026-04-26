from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


JsonDict = Dict[str, Any]


def _bucket_label_by_percent(percent: float) -> str:
    clipped = max(0.0, min(100.0, percent))
    if clipped >= 100.0:
        return "100%"
    left = int(clipped // 10) * 10
    right = left + 10
    return f"{left}-{right}%"


def _load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _record_ratio_bucket(rec: JsonDict, metric: str) -> str:
    nodes = rec.get("results", {}).get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return "invalid"

    if metric == "prove":
        prove_required_nodes = [
            node for node in nodes if int(node.get("needs_verification", 1)) != 0
        ]
        denom = len(prove_required_nodes)
        if denom <= 0:
            return "invalid"
        numer = sum(
            1
            for node in prove_required_nodes
            if bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        )
        return _bucket_label_by_percent((numer / denom) * 100.0)

    # metric == "form"
    denom = len(nodes)
    numer = sum(
        1 for node in nodes if bool((node.get("formalization") or {}).get("lean_pass", False))
    )
    return _bucket_label_by_percent((numer / denom) * 100.0)


def _build_id_to_bucket(stage3_jsonl: Path, metric: str) -> Dict[str, str]:
    rows = _load_jsonl(stage3_jsonl)
    result: Dict[str, str] = {}
    for rec in rows:
        rid = str(rec.get("meta", {}).get("record_id", "")).strip()
        if not rid:
            continue
        result[rid] = _record_ratio_bucket(rec, metric)
    return result


def _record_all_nodes_prove_verified(rec: JsonDict) -> bool:
    nodes = rec.get("results", {}).get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return False
    required_nodes = [node for node in nodes if int(node.get("needs_verification", 1)) != 0]
    if not required_nodes:
        return False
    return all(
        bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        for node in required_nodes
    )


def _record_final_answer_wrong(rec: JsonDict) -> bool:
    nodes = rec.get("results", {}).get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return False
    final_nodes_required = [
        node
        for node in nodes
        if str(node.get("role", "")).strip() == "final"
        and int(node.get("needs_verification", 1)) != 0
    ]
    if not final_nodes_required:
        return False
    return any(
        not bool((node.get("solved_lemma") or {}).get("lean_verify", False))
        for node in final_nodes_required
    )


def _final_answer_wrong_stats(stage3_jsonl: Path) -> JsonDict:
    rows = _load_jsonl(stage3_jsonl)
    valid_records = 0
    all_nodes_prove_verified_records = 0
    final_answer_wrong_records = 0

    for rec in rows:
        nodes = rec.get("results", {}).get("nodes", [])
        if not isinstance(nodes, list) or not nodes:
            continue
        valid_records += 1
        if _record_all_nodes_prove_verified(rec):
            all_nodes_prove_verified_records += 1
        if _record_final_answer_wrong(rec):
            final_answer_wrong_records += 1

    non_fully_verified_records = max(valid_records - all_nodes_prove_verified_records, 0)
    final_wrong_ratio_in_total = (
        final_answer_wrong_records / valid_records if valid_records else 0.0
    )
    final_wrong_ratio_in_non_fully_verified = (
        final_answer_wrong_records / non_fully_verified_records
        if non_fully_verified_records
        else 0.0
    )

    return {
        "valid_records_with_nodes": valid_records,
        "all_nodes_prove_verified_records": all_nodes_prove_verified_records,
        "non_fully_verified_records": non_fully_verified_records,
        "final_answer_wrong_records": final_answer_wrong_records,
        "final_answer_wrong_ratio_in_total": final_wrong_ratio_in_total,
        "final_answer_wrong_ratio_in_non_fully_verified": final_wrong_ratio_in_non_fully_verified,
    }


def _compare_two_experiments(
    exp_root: Path,
    exp_a: str,
    exp_b: str,
    metric: str,
    include_unchanged: bool,
) -> JsonDict:
    stage3_a = exp_root / exp_a / "result_stage3" / "stage3_results.jsonl"
    stage3_b = exp_root / exp_b / "result_stage3" / "stage3_results.jsonl"
    if not stage3_a.is_file():
        raise SystemExit(f"missing stage3 file for exp_a: {stage3_a}")
    if not stage3_b.is_file():
        raise SystemExit(f"missing stage3 file for exp_b: {stage3_b}")

    id_to_bucket_a = _build_id_to_bucket(stage3_a, metric=metric)
    id_to_bucket_b = _build_id_to_bucket(stage3_b, metric=metric)
    final_wrong_a = _final_answer_wrong_stats(stage3_a)
    final_wrong_b = _final_answer_wrong_stats(stage3_b)

    ids_a = set(id_to_bucket_a.keys())
    ids_b = set(id_to_bucket_b.keys())
    common_ids = sorted(ids_a & ids_b)
    only_in_a = sorted(ids_a - ids_b)
    only_in_b = sorted(ids_b - ids_a)

    flows: List[JsonDict] = []
    flow_counter: Counter[Tuple[str, str]] = Counter()
    changed_count = 0
    for rid in common_ids:
        from_bucket = id_to_bucket_a[rid]
        to_bucket = id_to_bucket_b[rid]
        changed = from_bucket != to_bucket
        if changed:
            changed_count += 1
        if changed or include_unchanged:
            flows.append(
                {
                    "record_id": rid,
                    "from_bucket": from_bucket,
                    "to_bucket": to_bucket,
                    "changed": changed,
                }
            )
        flow_counter[(from_bucket, to_bucket)] += 1

    flow_summary = [
        {"from_bucket": src, "to_bucket": dst, "count": cnt}
        for (src, dst), cnt in sorted(
            flow_counter.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]

    final_wrong_change = {
        "exp_a": final_wrong_a,
        "exp_b": final_wrong_b,
        "delta_final_answer_wrong_records": (
            final_wrong_b["final_answer_wrong_records"]
            - final_wrong_a["final_answer_wrong_records"]
        ),
        "delta_final_wrong_ratio_in_total": (
            final_wrong_b["final_answer_wrong_ratio_in_total"]
            - final_wrong_a["final_answer_wrong_ratio_in_total"]
        ),
        "delta_final_wrong_ratio_in_non_fully_verified": (
            final_wrong_b["final_answer_wrong_ratio_in_non_fully_verified"]
            - final_wrong_a["final_answer_wrong_ratio_in_non_fully_verified"]
        ),
    }

    return {
        "metric": metric,
        "exp_root": str(exp_root),
        "exp_a": exp_a,
        "exp_b": exp_b,
        "count_exp_a_ids": len(ids_a),
        "count_exp_b_ids": len(ids_b),
        "count_common_ids": len(common_ids),
        "count_only_in_a": len(only_in_a),
        "count_only_in_b": len(only_in_b),
        "count_changed_ids": changed_count,
        "changed_ratio_in_common": (changed_count / len(common_ids) if common_ids else 0.0),
        "flow_summary": flow_summary,
        "final_answer_wrong_change": final_wrong_change,
        "id_flows": flows,
        "only_in_a_ids": only_in_a,
        "only_in_b_ids": only_in_b,
    }


def _write_csv(path: Path, rows: List[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["record_id", "from_bucket", "to_bucket", "changed"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "输入两个实验名，比较同一 record_id 在两次实验中的分桶去向变化（A -> B）。"
        )
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="实验根目录，默认是仓库下 results/",
    )
    parser.add_argument("-a", "--exp-a", required=True, help="第一次实验名（作为 from）")
    parser.add_argument("-b", "--exp-b", required=True, help="第二次实验名（作为 to）")
    parser.add_argument(
        "--metric",
        choices=("prove", "form"),
        default="prove",
        help="比较 prove 或 form 分布桶，默认 prove",
    )
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="输出明细时包含未变化 id（默认只输出变化的 id）",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="结果 JSON 路径；默认输出到 <exp-root>/<exp-b>/stats/id_flow_<exp-a>_to_<exp-b>.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="id 明细 CSV 路径；默认输出到 <exp-root>/<exp-b>/stats/id_flow_<exp-a>_to_<exp-b>.csv",
    )
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    payload = _compare_two_experiments(
        exp_root=exp_root,
        exp_a=args.exp_a,
        exp_b=args.exp_b,
        metric=args.metric,
        include_unchanged=bool(args.include_unchanged),
    )

    default_name = f"id_flow_{args.exp_a}_to_{args.exp_b}"
    default_stats_dir = exp_root / args.exp_b / "stats"
    out_json = args.out_json or (default_stats_dir / f"{default_name}.json")
    out_csv = args.out_csv or (default_stats_dir / f"{default_name}.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, payload["id_flows"])

    print(f"metric: {payload['metric']}")
    print(f"exp_a -> exp_b: {payload['exp_a']} -> {payload['exp_b']}")
    print(f"common_ids: {payload['count_common_ids']}")
    print(f"changed_ids: {payload['count_changed_ids']}")
    print(f"changed_ratio_in_common: {payload['changed_ratio_in_common']:.4f}")
    print(f"only_in_a: {payload['count_only_in_a']}")
    print(f"only_in_b: {payload['count_only_in_b']}")
    final_wrong_change = payload["final_answer_wrong_change"]
    exp_a_final = final_wrong_change["exp_a"]
    exp_b_final = final_wrong_change["exp_b"]
    print("final_answer_wrong_records change:")
    print(
        "  absolute: "
        f"{exp_a_final['final_answer_wrong_records']} -> {exp_b_final['final_answer_wrong_records']} "
        f"(delta={final_wrong_change['delta_final_answer_wrong_records']})"
    )
    print(
        "  ratio_in_total: "
        f"{exp_a_final['final_answer_wrong_ratio_in_total']:.4f} -> "
        f"{exp_b_final['final_answer_wrong_ratio_in_total']:.4f} "
        f"(delta={final_wrong_change['delta_final_wrong_ratio_in_total']:.4f})"
    )
    print(
        "  ratio_in_non_fully_verified: "
        f"{exp_a_final['final_answer_wrong_ratio_in_non_fully_verified']:.4f} -> "
        f"{exp_b_final['final_answer_wrong_ratio_in_non_fully_verified']:.4f} "
        f"(delta={final_wrong_change['delta_final_wrong_ratio_in_non_fully_verified']:.4f})"
    )
    print("top flows:")
    for item in payload["flow_summary"][:10]:
        print(f"  {item['from_bucket']} -> {item['to_bucket']}: {item['count']}")
    print(f"json saved to: {out_json}")
    print(f"csv  saved to: {out_csv}")


if __name__ == "__main__":
    main()
