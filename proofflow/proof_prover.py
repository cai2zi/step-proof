import re

import httpx

from .lean_check import LeanServer, process_lean_string
from .node_schema import is_condition, is_context
from .proof_graph import Definition, TheoremCondition
from .prompt_builder import TaskProfile, build_chat_messages
from .utils import LLMManager


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
    except Exception as e:
        raise RuntimeError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "lean_verify": lean_verify,
        "error_msg": error_msg,
    }


def run_solver_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    task_profile: TaskProfile = "proof",
    logs=None,
    max_retries: int = 3,
    prove_negation: bool = False,
    enable_ctx_solver: bool = True,
) -> tuple:
    """
    Builds prompts from prompts/{proof|calc}/** templates and calls the LLM API.
    """

    node_type = getattr(item, "node_type", None)
    needs_verification = int(getattr(item, "needs_verification", 0) or 0)
    is_condition_or_def = isinstance(item, TheoremCondition) or isinstance(
        item, Definition
    )
    if is_condition_or_def:
        if not enable_ctx_solver:
            return {}
        if is_condition(item.id, node_type):
            return {}
        if is_context(item.id, node_type) and needs_verification != 1:
            return {}
    if not item.formalization["lean_code"]:
        return {}

    stage = "prove_negation" if prove_negation else "prove"
    messages = build_chat_messages(
        task_profile,
        stage,
        statement=item.statement,
        lean_code=item.formalization["lean_code"],
    )
    if not item.formalization["lean_pass"]:
        messages[1]["content"] += (
            "\n\nThe previous Lean4 code I sent you contains errors. Please take that into account."
        )

    results = {
        "lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": "failure to get any response from LLM",
    }

    attempt_history = []
    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        try:
            results = extract_code_validate(response, lean_server)
            results["tries"] = attempt + 1
            results["attempt_history"] = attempt_history

            if results["lean_verify"]:
                return results
            messages.append(
                {
                    "role": "user",
                    "content": f"Lean error/warnings: "
                    + str(results["error_msg"])
                    + "\n\n Based on these errors, please correct the previous response. ",
                }
            )
        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\n Based on these errors, please correct the previous response. ",
                }
            )

        attempt_history.append(results)

    return results
