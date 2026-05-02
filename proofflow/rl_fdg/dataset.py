from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .prompting import build_builder_prompt_messages


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class FDGRLExample:
    record_id: str
    problem_text: str
    solution_or_cot: str
    prompt: List[JsonDict]
    reference_answer: str = ""
    data_source: str = "fdg_builder"
    split: str = "train"
    extra_info: JsonDict = field(default_factory=dict)

    def to_verl_record(self) -> JsonDict:
        reward_ground_truth = self.reference_answer or ""
        return {
            "data_source": self.data_source,
            "prompt": self.prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": reward_ground_truth},
            "extra_info": {
                "record_id": self.record_id,
                "problem_text": self.problem_text,
                "solution_or_cot": self.solution_or_cot,
                "reference_answer": self.reference_answer,
                "split": self.split,
                **dict(self.extra_info),
            },
        }


def iter_parquet_examples(
    *,
    parquet_dir: Path,
    glob_pattern: str,
    id_column: str,
    question_column: str,
    response_column: str,
    answer_column: Optional[str] = None,
    limit: int = -1,
) -> Iterator[JsonDict]:
    loaded = 0
    for fp in sorted(parquet_dir.glob(glob_pattern)):
        if not fp.is_file():
            continue
        df = pd.read_parquet(fp)
        required = [id_column, question_column, response_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{fp} is missing required columns: {missing}")
        for row_pos, (_, row) in enumerate(df.iterrows()):
            yield {
                "record_id": str(row[id_column]),
                "problem_text": str(row[question_column]),
                "solution_or_cot": str(row[response_column]),
                "reference_answer": str(row[answer_column]) if answer_column and answer_column in row else "",
                "source_file": str(fp),
                "source_row_pos": row_pos,
            }
            loaded += 1
            if limit >= 0 and loaded >= limit:
                return


def build_rl_examples(
    records: Iterable[JsonDict],
    *,
    prompt_name: str,
    include_think_in_dag: bool,
    data_source: str,
    split: str,
) -> List[FDGRLExample]:
    examples: List[FDGRLExample] = []
    for row in records:
        prompt = build_builder_prompt_messages(
            problem_text=str(row["problem_text"]),
            solution_or_cot=str(row["solution_or_cot"]),
            prompt_name=prompt_name,
            include_think_in_dag=include_think_in_dag,
        )
        examples.append(
            FDGRLExample(
                record_id=str(row["record_id"]),
                problem_text=str(row["problem_text"]),
                solution_or_cot=str(row["solution_or_cot"]),
                prompt=prompt,
                reference_answer=str(row.get("reference_answer", "")),
                data_source=data_source,
                split=split,
                extra_info={
                    "source_file": str(row.get("source_file", "")),
                    "source_row_pos": int(row.get("source_row_pos", -1)),
                    "fdg_prompt": prompt_name,
                    "include_think_in_dag": bool(include_think_in_dag),
                },
            )
        )
    return examples


def write_verl_parquet(records: List[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def write_json_manifest(payload: JsonDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
