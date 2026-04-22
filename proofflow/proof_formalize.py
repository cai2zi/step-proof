import re
from typing import List

import httpx

from .lean_check import LeanServer, process_lean_string
from .node_schema import is_condition, is_context, is_final
from .prompt_builder import TaskProfile, build_chat_messages
from .utils import LLMManager, remove_imports


def extract_code_validate(text_input, lean_server):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r"```lean4\n(.*?)\n```", text_input, re.DOTALL)
        if not matches:
            raise ValueError("No Lean 4 code block found.")
    except RuntimeError as e:
        return f"Error during code extraction: {str(e)}. Is ```lean4 ``` written?"

    response = matches[-1].strip()
    response = process_lean_string(response)  # add missing imports

    try:
        lean_pass, lean_verify, error_msg = lean_server.check_lean_string(response)
    except RuntimeError as e:
        raise ValueError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "error_msg": [] if lean_pass else error_msg,
    }


def _build_dependency_sections(
    task_profile: TaskProfile,
    all_items: list,
    dependencies: List[str],
    previous_context: bool,
) -> tuple[str, str]:
    """
    Returns (dependency_lean_code, dependency_context_block) for templates.
    
    """
    if not previous_context or not all_items or not dependencies:
        return "", ""

    intro = (
        f"\n\n This proof step depend on previous proof steps, namely steps {dependencies}.\n"
        "Please make use use of their formal lean4 code, which contains relevant lean4 hypothesis and type declarations you may use:"
    )
    parts: List[str] = []
    for d in all_items:
        if d.id in dependencies:
            parts.append("\n")
            if (
                hasattr(d, "formalization")
                and d.formalization
                and d.formalization.get("lean_code")
                and d.formalization.get("lean_pass")
            ):
                parts.append(remove_imports(d.formalization["lean_code"]))
            else:
                parts.append(
                    f"Dependency step {d.id} is provided in natural language: \"{d.statement}\". "
                    f"Please formalize it as part of your current lemma's hypotheses."
                )
                
    if not parts:
        return "", ""

    footer = (
        "\nFocus on the original formalization task I gave you and use the previous Lean codes, extra context, type declarations, variables domains, etc. You can assume the information is correct. Make use of it!"
    )
    combined = "\n".join(parts)

    return "", (intro + combined + footer).strip()


def _build_original_proof_block(original_proof: str, supply_proof: bool) -> str:
    if not supply_proof or not original_proof:
        return ""
    return (
        "\n\n This formalization task is a proof step which is part of a larger full proof given next:\n"
        + original_proof
        + "\nThe full proof may contain extra missing information that you need, specially variable types and domains (e.g 'r' is real and positive). Make use of it, specially if you encounter errors."
        "\nHowever, please focus on the original formalization task I gave you and use the previous full proof for extra context only."
    )


def run_formalizer_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    task_profile: TaskProfile = "proof",
    all_items: list = None,
    logs=None,
    max_retries: int = 3,
    previous_context: bool = True,
    original_proof: str = "",
    supply_proof: bool = True,
) -> tuple:
    """
    Builds prompts from prompts/{proof|calc}/** templates and calls the LLM API.
    """
    all_items = all_items or []
    node_type = getattr(item, "node_type", None)
    needs_verification = int(getattr(item, "needs_verification", 0) or 0)
    is_condition_or_def = is_condition(item.id, node_type) or (
        is_context(item.id, node_type) and needs_verification != 1
    )
    if is_final(item.id, node_type):
        lemma_header = f"theorem {item.id}"
    else:
        lemma_header = f"lemma {item.id}"
    dependencies = item.dependencies

    dep_lean, dep_ctx = _build_dependency_sections(
        task_profile, all_items, dependencies, previous_context
    )
    original_block = _build_original_proof_block(original_proof, supply_proof)

    if is_condition_or_def: # 对于pc, ctx 不需要verify 这部分的message构建
        kwargs = {
            "node_id": item.id,
            "statement": item.statement,
            "dependency_context_block": dep_ctx or "",
        }
        if task_profile != "calc":
            kwargs["original_proof_block"] = original_block
        messages = build_chat_messages(task_profile, "formalize_context", **kwargs)
    else:
        kwargs = {
            "lemma_header": lemma_header,
            "statement": item.statement,
            "dependencies": dependencies,
            "dependency_context_block": dep_ctx or "",
        }
        if task_profile != "calc":
            kwargs["original_proof_block"] = original_block
        messages = build_chat_messages(task_profile, "formalize_claim", **kwargs)

    formalization = {
        "lean_code": "",
        "lean_pass": False,
        "error_msg": "Failure to get a valid formalization from the LLM",
    }

    attempt_history = []

    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        try:
            formalization = extract_code_validate(response, lean_server)
            formalization["tries"] = attempt + 1
            formalization["attempt_history"] = attempt_history

            if formalization["lean_pass"]:
                return formalization
            messages.append(
                {
                    "role": "user",
                    "content": f"Lean error: "
                    + str(formalization["error_msg"])
                    + "\n\nBased on the error, please correct the previous response. ",
                }
            )

        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\nBased on the error, please correct the previous response. ",
                }
            )

        attempt_history.append(formalization)

    return formalization
