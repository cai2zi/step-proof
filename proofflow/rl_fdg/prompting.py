from __future__ import annotations

from typing import Dict, List

from proofflow.fdg_graph import build_fdg_messages


def build_builder_prompt_messages(
    *,
    problem_text: str,
    solution_or_cot: str,
    prompt_name: str = "fdg_origin4",
    include_think_in_dag: bool = False,
) -> List[Dict[str, str]]:
    return build_fdg_messages(
        problem_text=problem_text,
        solution_or_cot=solution_or_cot,
        include_think_in_dag=include_think_in_dag,
        prompt_name=prompt_name,
    )
