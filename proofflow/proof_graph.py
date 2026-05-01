import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json5
from pydantic import BaseModel, Field, ValidationError

from .node_schema import (
    ROLE_CLAIM,
    ROLE_CONDITION,
    ROLE_CONTEXT,
    ROLE_FINAL,
    infer_role,
    is_final,
    is_structural_final,
    normalize_nodes,
)
from .prompt_builder import TaskProfile, build_chat_messages
from .utils import LLMManager


class BaseTheoremComponent(BaseModel):
    id: str
    natural_language: str = Field(..., description="Exact text from theorem/proof")
    statement: str = Field(..., description="Self contained NL statement")
    node_type: Optional[str] = Field(default=None, description="Semantic node type")
    needs_verification: Optional[int] = Field(
        default=None, description="Whether node should be verified"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of dependency IDs"
    )
    formalization: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from formalization process"
    )
    solved_lemma: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process"
    )
    score: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from scorer process"
    )
    error_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from error report"
    )


class TheoremCondition(BaseTheoremComponent):
    pass


class Definition(BaseTheoremComponent):
    pass


class Lemma(BaseTheoremComponent):
    # lean_hint: str = Field(..., description="Hint for proving this step")
    solved_negation: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process of the negation."
    )


class TheoremStatement(BaseTheoremComponent):
    # lean_hint: str = Field(..., description="Final step hint")
    solved_negation: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process of the negation."
    )

ProofGraphItem = TheoremCondition | Definition | Lemma | TheoremStatement


def check_DAG(
    validated_graph: List[ProofGraphItem],
    id_schema_mode: str = "auto",
    validation_profile: str = "strict",
) -> List[ProofGraphItem]:
    """
    Checks and fixes the Directed Acyclic Graph (DAG) structure of the proof.

    1. Checks if lemmas are used as dependencies (warns if they're orphaned)
    2. checks if graph as cycles and forward references

    Args:
        validated_graph: List of validated proof graph items

    Returns:
        List of proof graph items with fixed dependencies
    """
    # Create a mapping of all item IDs for quick lookup
    all_ids = {item.id for item in validated_graph}

    # Track which lemma IDs are used as dependencies
    used_lemma_ids = set()

    # First pass: collect all dependencies to find used lemmas
    for item in validated_graph:
        used_lemma_ids.update(item.dependencies)

    # Second pass: check for orphaned non-final nodes (fa_/ts_ prefix always counts as final)
    for item in validated_graph:
        if (
            not is_structural_final(item.id, id_schema_mode)
            and not is_final(item.id, getattr(item, "node_type", None), mode=id_schema_mode)
            and item.id not in used_lemma_ids
        ):
            error_msg = (
                f"'{item.id}' is not used as a dependency "
                f"by any subsequent lemma or theorem statement. "
                f"'{item.id}' needs to be used somewhere so that we have a valid proof."
                f"Please reconsider the graph structure."
            )
            print("Error: ", error_msg)
            raise ValueError(error_msg)

    # Third pass: validate DAG structure (check for cycles)
    def has_cycle(graph_items: List[ProofGraphItem]) -> bool:
        """Check if the dependency graph has any cycles."""
        # Build adjacency list
        adj_list = {item.id: item.dependencies for item in graph_items}

        # Track visit states: 0 = unvisited, 1 = visiting, 2 = visited
        visit_state = {item_id: 0 for item_id in all_ids}

        def dfs(node_id: str) -> bool:
            """Depth-first search to detect cycles."""
            if node_id not in adj_list:
                return False

            if visit_state[node_id] == 1:  # Currently visiting = cycle found
                return True
            if visit_state[node_id] == 2:  # Already visited
                return False

            visit_state[node_id] = 1  # Mark as visiting

            # Check all dependencies
            for dep_id in adj_list[node_id]:
                if dep_id in visit_state and dfs(dep_id):
                    return True

            visit_state[node_id] = 2  # Mark as visited
            return False

        # Check each node
        for item_id in all_ids:
            if visit_state[item_id] == 0:
                if dfs(item_id):
                    return True
        return False

    # Check for cycles
    if has_cycle(validated_graph):
        error_message = (
            "Cycle detected in the dependency graph! "
            "This violates the DAG property and will cause issues in proof formalization."
        )
        print("Error: ", error_message)
        raise ValueError(error_message)

    # Additional validation: check for forward references
    for i, item in enumerate(validated_graph):
        available_ids = {validated_graph[j].id for j in range(i)}

        for dep_id in item.dependencies:
            if dep_id not in available_ids and dep_id not in all_ids:
                msg = f"Item '{item.id}' references unknown dependency '{dep_id}'"
                if validation_profile == "strict":
                    raise ValueError(msg)
                print("Warning:", msg)
            elif dep_id not in available_ids:
                # This is a forward reference
                error_msg = (
                    f"Item '{item.id}' has a forward reference to '{dep_id}'. "
                    f"Dependencies should only reference previous items in the proof."
                )
                if validation_profile == "strict":
                    print("Error: ", error_msg)
                    raise ValueError(error_msg)
                print("Warning:", error_msg)

    return validated_graph


def validate_proof_graph(
    data: List[dict],
    id_schema_mode: str = "auto",
    validation_profile: str = "strict",
) -> List[ProofGraphItem]:
    """
    Validates the proof graph data using Pydantic models.
    Returns the validated data or raises ValidationError.
    """
    if not data:
        raise ValueError("The parsed graph is empty (no nodes). Please generate at least one valid node.")

    validated_items = []
    normalized_data = normalize_nodes(data)
    for item in normalized_data:
        role = infer_role(item.get("id", ""), item.get("node_type"), mode=id_schema_mode)
        if role == ROLE_FINAL:
            validated_items.append(TheoremStatement(**item))
        elif role == ROLE_CLAIM:
            validated_items.append(Lemma(**item))
        elif role == ROLE_CONTEXT:
            validated_items.append(Definition(**item))
        elif role == ROLE_CONDITION:
            validated_items.append(TheoremCondition(**item))
        else:
            raise ValueError(
                f"Unknown node type/id: {item.get('id')} (node_type={item.get('node_type')})."
                " Expected calc prefixes pc_/ctx_/c_/fa_ or legacy prefixes tc_/def_/l*/ts_."
            )
    return check_DAG(
        validated_items,
        id_schema_mode=id_schema_mode,
        validation_profile=validation_profile,
    )


def extract_json_block(text: str):
    """Extract JSON block with multiple fallback strategies."""
    
    # Strategy 0: Remove <think>...</think> blocks if present
    if '</think>' in text:
        text = text.split('</think>')[-1]
    else:
        text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if text_no_think.strip():
            text = text_no_think
    
    # Strategy 1: Look for fenced blocks (more permissive)
    patterns = [
        r"```json\s*\n?(.*?)\n?```",           # ```json ... ```
        r"```\s*\n?(.*?)\n?```",               # ``` ... ```  (no language)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            # Prefer the last match that is a list, as DAG payload must be a list
            for candidate in reversed(matches):
                candidate = candidate.strip()
                if candidate.startswith('['):
                    return candidate
            for candidate in reversed(matches):
                candidate = candidate.strip()
                if candidate.startswith('{'):
                    return candidate
    
    # Strategy 2: Find balanced JSON array or object
    def find_balanced(s: str, open_char: str, close_char: str):
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(s):
            if char == open_char:
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == close_char and start_idx != -1:
                brace_count -= 1
                if brace_count == 0:
                    return s[start_idx:i+1]
        return None

    # First try to find an array (DAG is usually a list)
    arr_match = find_balanced(text, '[', ']')
    if arr_match:
        return arr_match
        
    # Fallback to an object
    obj_match = find_balanced(text, '{', '}')
    if obj_match:
        return obj_match
    
    # Strategy 3: Look for any JSON-like structure
    json_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(0)
        
    raise ValueError(f"No JSON block found. Text starts with: {repr(text[:100])}")

def sanitize_backslashes(raw: str) -> str:
    """
    Fix invalid JSON escape sequences by doubling backslashes.
    This version assumes any non-standard backslash is a single character
    that needs to be escaped. It correctly handles the LaTeX-style inputs.
    """

    # Replace common LaTeX sequences with their escaped equivalents
    raw = raw.replace('\\\\', '\\')  # Normalize any existing double backslashes
    raw = raw.replace('\\in', '\\\\in')
    raw = raw.replace('\\Q', '\\\\Q')
    # Let's assume the common LLM error is unescaped LaTeX commands.
    raw = raw.replace(r'\int', r'\\int')
    raw = raw.replace(r'\mathbb{Q}', r'\\mathbb{Q}')
    
    return raw.replace('\\', '\\\\')

def parse_llm_json(content: str):
    """Enhanced version with better debugging."""
    try:
        raw = extract_json_block(content)        
        # Try parsing without sanitization first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Only sanitize if regular parsing fails
            sanitized = sanitize_backslashes(raw)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                return json5.loads(sanitized)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with open("failed_llm_json.txt", "w") as f:
                f.write(content)
            print("Wrote failed LLM output to failed_llm_json.txt")
        except:
            pass
        print(f"DEBUG: Full content: {repr(content[:200])}...")  # Remove this in production
        raise ValueError(f"Failed to parse JSON: {e}") from e


def condition_on_all_previous_steps(graph_info: List[ProofGraphItem]):
    passed_lemmas = []
    for item in graph_info:
        item.dependencies = passed_lemmas.copy()  # Use copy to avoid reference issues
        passed_lemmas.append(item.id)


def drop_orphan_nodes(data: List[dict], id_schema_mode: str) -> List[dict]:
    """
    Remove non-final nodes that never appear in any node's dependencies (orphans).
    Repeat until no such nodes remain; strip dangling dependency ids after each pass.
    """
    data = deepcopy(data)
    changed = True
    while changed:
        changed = False
        used: set[str] = set()
        for n in data:
            used.update(n.get("dependencies") or [])
        remove_ids = {
            n["id"]
            for n in data
            if not is_structural_final(n["id"], id_schema_mode)
            and not is_final(n["id"], n.get("node_type"), mode=id_schema_mode)
            and n["id"] not in used
        }
        if remove_ids:
            changed = True
            data = [n for n in data if n["id"] not in remove_ids]
            remaining = {n["id"] for n in data}
            for n in data:
                n["dependencies"] = [
                    d for d in (n.get("dependencies") or []) if d in remaining
                ]
    return data


# ---------------------------------------------------------------------------
# New pure functions for batch-based Stage-1 pipeline
# ---------------------------------------------------------------------------

@dataclass
class GraphParseResult:
    """Result of a single parse-and-validate attempt on one LLM response."""
    ok: bool
    items: Optional[List["ProofGraphItem"]]
    error_msg: Optional[str]
    last_parsed_graph: Optional[List[dict]]


def strip_think_blocks(text: str) -> str:
    """Remove model-private <think>...</think> spans before DAG extraction."""
    without_closed = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think\b[^>]*>.*\Z", "", without_closed, flags=re.DOTALL | re.IGNORECASE).strip()


def build_graph_messages(
    task_profile: TaskProfile,
    problem: str = "",
    raw_cot: str = "",
    natural_language_proof: str = "",
    include_think_in_dag: bool = True,
) -> List[Dict[str, str]]:
    """Build the initial messages list for the graph LLM (no LLM call)."""
    if task_profile == "calc":
        if not problem or not raw_cot:
            raise ValueError("task_profile='calc' requires non-empty problem and raw_cot")
        dag_raw_cot = raw_cot if include_think_in_dag else strip_think_blocks(raw_cot)
        return build_chat_messages("calc", "dag", PROBLEM=problem, RAW_COT=dag_raw_cot)
    elif task_profile == "proof":
        if not natural_language_proof:
            raise ValueError("task_profile='proof' requires natural_language_proof")
        return build_chat_messages("proof", "dag", NATURAL_LANGUAGE_PROOF=natural_language_proof)
    else:
        raise ValueError(f"Unknown task_profile: {task_profile}")


def parse_and_validate_graph(
    content: str,
    id_schema_mode: str = "calc",
    validation_profile: str = "strict",
    follow_dag: bool = True,
    attempt: int = 0,
    allow_graph_rewrite_after: int = 3,
) -> GraphParseResult:
    """Parse and validate a single LLM response string.

    Returns a GraphParseResult; never raises. Caller inspects .ok and .error_msg.
    """
    last_parsed_graph: Optional[List[dict]] = None

    try:
        proof_graph_raw = parse_llm_json(content)
        proof_graph = normalize_nodes(proof_graph_raw)
        last_parsed_graph = proof_graph
    except Exception as e:
        return GraphParseResult(
            ok=False,
            items=None,
            error_msg="Could not parse proof graph JSON from the LLM output. Error: " + str(e),
            last_parsed_graph=None,
        )

    try:
        validated_graph = validate_proof_graph(
            proof_graph,
            id_schema_mode=id_schema_mode,
            validation_profile=validation_profile,
        )
        if not follow_dag:
            condition_on_all_previous_steps(validated_graph)
        return GraphParseResult(
            ok=True, items=validated_graph, error_msg=None, last_parsed_graph=last_parsed_graph
        )
    except (ValidationError, ValueError) as e:
        if attempt + 1 >= allow_graph_rewrite_after:
            error_msg = (
                f"JSON validation failed. You may restructure the DAG if needed, "
                f"but keep it faithful to the source text and fix these errors first: {str(e)}."
            )
        else:
            error_msg = (
                "JSON validation failed. Keep existing IDs and node types unchanged; "
                f"only fix JSON/schema/dependency errors: {str(e)}."
            )
        print(str(e))
        return GraphParseResult(
            ok=False, items=None, error_msg=error_msg, last_parsed_graph=last_parsed_graph
        )


def append_error_to_messages(
    messages: List[Dict[str, str]],
    error_msg: str,
) -> List[Dict[str, str]]:
    """Append LLM assistant reply + error feedback to message history for retry."""
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": error_msg})
    return messages


def try_orphan_drop_recovery(
    last_parsed_graph: List[dict],
    id_schema_mode: str,
    validation_profile: str,
    follow_dag: bool,
    max_retries: int,
) -> Optional[List["ProofGraphItem"]]:
    """Attempt orphan-drop cleanup after all retries are exhausted.

    Returns validated items on success, None on failure.
    """
    try:
        cleaned = drop_orphan_nodes(last_parsed_graph, id_schema_mode)
        if not cleaned:
            ids = [n.get("id") for n in last_parsed_graph]
            raise ValueError(
                "Graph became empty after dropping orphan nodes. "
                f"No fa_/ts_-style final node was kept; raw ids were: {ids}. "
                "Ensure at least one final node (fa_* or ts_*) or fix dependencies."
            )
        validated_graph = validate_proof_graph(
            cleaned,
            id_schema_mode=id_schema_mode,
            validation_profile=validation_profile,
        )
        if not follow_dag:
            condition_on_all_previous_steps(validated_graph)
        print(
            f"Warning: DAG had orphan nodes; removed them and accepted the graph "
            f"after {max_retries} failed validation attempt(s)."
        )
        return validated_graph
    except (ValidationError, ValueError) as e:
        print(f"Sanitized graph still invalid: {e}")
        return None


# ---------------------------------------------------------------------------
# Legacy single-record function – kept for ProofFlow (Stage 2) compatibility
# ---------------------------------------------------------------------------

def build_proof_graph(
    model_manager: LLMManager,
    task_profile: TaskProfile = "proof",
    problem: str = "",
    raw_cot: str = "",
    natural_language_proof: str = "",
    logs=None,
    follow_dag: bool = True,
    max_retries: int = 3,
    id_schema_mode: str = "auto",
    validation_profile: str = "strict",
    allow_graph_rewrite_after: int = 3,
    include_think_in_dag: bool = True,
) -> tuple[List[ProofGraphItem], int]:
    """
    Calls the graph LLM using prompts/{proof|calc}/system/dag.md + user/dag.md.
    Returns the parsed and validated JSON proof graph as a list of Pydantic models.
    """

    if task_profile == "calc":
        if not problem or not raw_cot:
            raise ValueError("task_profile='calc' requires non-empty problem and raw_cot")
        dag_raw_cot = raw_cot if include_think_in_dag else strip_think_blocks(raw_cot)
        messages = build_chat_messages("calc", "dag", PROBLEM=problem, RAW_COT=dag_raw_cot)
    elif task_profile == "proof":
        if not natural_language_proof:
            raise ValueError("task_profile='proof' requires natural_language_proof")
        messages = build_chat_messages(
            "proof", "dag", NATURAL_LANGUAGE_PROOF=natural_language_proof
        )
    else:
        raise ValueError(f"Unknown task_profile: {task_profile}")

    last_parsed_graph: Optional[List[dict]] = None
    for attempt in range(max_retries):
        content, messages = model_manager.call_llm(messages, logs=logs)
        # Try to parse JSON
        try:
            proof_graph_raw = parse_llm_json(content)
            with open("success_llm_json.txt", "w") as f:
                f.write(content)
            proof_graph = normalize_nodes(proof_graph_raw)
            last_parsed_graph = proof_graph
        except Exception as e:
            error_msg = "Could not parse proof graph JSON from the LLM output."
            messages.append(
                {"role": "user", "content": error_msg + ". Error: " + str(e)}
            )
            continue
        # Validate the parsed data
        try:
            validated_graph = validate_proof_graph(
                proof_graph,
                id_schema_mode=id_schema_mode,
                validation_profile=validation_profile,
            )

            if not follow_dag:  # condition on all previous steps
                condition_on_all_previous_steps(validated_graph)
            return validated_graph, attempt + 1

        except (ValidationError, ValueError) as e:
            if attempt + 1 >= allow_graph_rewrite_after:
                error_msg = (
                    f"JSON validation failed. You may restructure the DAG if needed, "
                    f"but keep it faithful to the source text and fix these errors first: {str(e)}."
                )
            else:
                error_msg = (
                    "JSON validation failed. Keep existing IDs and node types unchanged; "
                    f"only fix JSON/schema/dependency errors: {str(e)}."
                )
            messages.append({"role": "user", "content": error_msg})

            print(str(e))

    # After max retries: drop orphan non-final nodes and accept if the graph validates.
    if last_parsed_graph is not None:
        try:
            cleaned = drop_orphan_nodes(last_parsed_graph, id_schema_mode)
            if not cleaned:
                ids = [n.get("id") for n in last_parsed_graph]
                raise ValueError(
                    "Graph became empty after dropping orphan nodes. "
                    f"No fa_/ts_-style final node was kept; raw ids were: {ids}. "
                    "Ensure at least one final node (fa_* or ts_*) or fix dependencies."
                )
            validated_graph = validate_proof_graph(
                cleaned,
                id_schema_mode=id_schema_mode,
                validation_profile=validation_profile,
            )
            if not follow_dag:
                condition_on_all_previous_steps(validated_graph)
            print(
                "Warning: DAG had orphan nodes; removed them and accepted the graph "
                f"after {max_retries} failed validation attempt(s)."
            )
            return validated_graph, max_retries
        except (ValidationError, ValueError) as e:
            print(f"Sanitized graph still invalid: {e}")

    raise RuntimeError(
        f"Failed to build a valid proof graph after {max_retries} attempt(s) "
        "(orphan-drop recovery also failed or no parse succeeded)."
    )
