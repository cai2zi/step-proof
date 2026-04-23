from typing import Any, Dict, List, Optional

from .lean_check import LeanServer
from .node_schema import ROLE_CLAIM, ROLE_CONDITION, ROLE_FINAL, infer_role, is_condition, is_context
from .prompt_builder import TaskProfile
from .proof_formalize import run_formalizer_prompt
from .proof_graph import build_proof_graph
from .proof_prover import run_solver_prompt
from .utils import LLMManager, remove_imports
from .vis import (
    build_dag,
    create_interactive_visualization,
    create_static_visualization,
)


class ProofFlow:
    """
    Main class for automated mathematical proof formalization using Large Language Models.
    
    ProofFlow converts natural language mathematical proofs into formalized Lean 4 code
    through a multi-step process involving proof graph generation, formalization,
    and automated proof generation.
    
    Attributes:
        lean_server (LeanServer): Server for Lean 4 verification
        graph_model_manager (LLMManager): LLM for proof graph generation
        formalize_model_manager (LLMManager): LLM for natural language formalization
        solver_model_manager (LLMManager): LLM for automated proof generation
        score_model_manager (Optional[LLMManager]): LLM for semantic scoring
        verbose (bool): Whether to print progress information
        proof_items (Optional[List]): Generated proof graph items
        nl_proof (Optional[str]): Original natural language proof
        llm_call_logs (List[Dict]): Logs of all LLM API calls
    """
    def __init__(self, 
                 lean_server: LeanServer,
                 graph_model_manager: LLMManager,
                 formalize_model_manager: LLMManager,
                 solver_model_manager: LLMManager,
                 score_model_manager: Optional[LLMManager] = None,
                 verbose: bool = True,
                 task_profile: TaskProfile = "proof",
                 id_schema_mode: Optional[str] = None,
                 validation_profile: str = "strict",
                 allow_graph_rewrite_after: int = 3,
                 enable_ctx_solver: bool = True):
        """
        Initialize ProofFlow with required components.
        
        Args:
            lean_server: Server for Lean 4 verification
            graph_model_manager: LLM manager for proof graph generation
            formalize_model_manager: LLM manager for natural language formalization
            solver_model_manager: LLM manager for automated proof generation
            score_model_manager: Optional LLM manager for semantic scoring
            verbose: Whether to print progress information during processing
            task_profile: "proof" (theorem+proof NL) or "calc" (problem + raw CoT).
            id_schema_mode: Override node-id convention; default maps calc→calc schema, else legacy.
        """
        self.proof_items = None
        self.nl_proof = None
        self.verbose = verbose
        self.graph_model_manager = graph_model_manager
        self.formalize_model_manager = formalize_model_manager 
        self.solver_model_manager = solver_model_manager 
        self.score_model_manager = score_model_manager 
        self.llm_call_logs = []  # Store LLM call logs per instance
        self.lean_server = lean_server
        self.task_profile = task_profile
        self.id_schema_mode = id_schema_mode or (
            "calc" if task_profile == "calc" else "legacy"
        )
        self.validation_profile = validation_profile
        self.allow_graph_rewrite_after = allow_graph_rewrite_after
        self.enable_ctx_solver = enable_ctx_solver
    
    # ===== PRIVATE UTILITIES =====
    
    def _print_status(self, msg: str, style: Optional[str] = None) -> None:
        """
        Print status message with optional styling if verbose mode is enabled.
        
        Args:
            msg: Message to print
            style: Optional style name ('header', 'okblue', 'okgreen', 'warning', 'fail', 'bold', 'underline')
        """
        if not self.verbose:
            return
        # ANSI color codes for style
        styles = {
            'header': '\033[95m',
            'okblue': '\033[94m',
            'okgreen': '\033[92m',
            'warning': '\033[93m',
            'fail': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m',
        }
        if style and style in styles:
            print(f"{styles[style]}{msg}{styles['end']}")
        else:
            print(msg)

    def _print_progress_summary(self, tries: int) -> None:
        """
        Print summary of proof graph generation progress.
        
        Args:
            tries: Number of attempts made during graph generation
        """
        n_conditions = 0
        n_lemmas = 0
        n_solutions = 0
        for item in self.proof_items:
            role = infer_role(item.id, getattr(item, "node_type", None), mode=self.id_schema_mode)
            if role == ROLE_CONDITION:
                n_conditions += 1
            elif role == ROLE_CLAIM:
                n_lemmas += 1
            elif role == ROLE_FINAL:
                n_solutions += 1
        if tries:
            self._print_status(f"\nProof graph completed ({tries} tries): {n_conditions} condition(s), {n_lemmas} lemma(s), {n_solutions} theorem solution(s).", style='header')
        else:
            self._print_status(f"{n_conditions} condition(s), {n_lemmas} lemma(s), {n_solutions} theorem solution(s).", style='header')

    def _print_item_progress(self, idx: int, total: int, item_id: str, completed: bool = False) -> None:
        """
        Print progress for individual proof item processing.
        
        Args:
            idx: Current item index (0-based)
            total: Total number of items
            item_id: Unique identifier for the item
            completed: Whether the item has been completed
        """
        if completed:
            self._print_status(f"   \u2714 Completed item {item_id}.", style='okgreen')
        else:
            self._print_status(f"\u27A4 Formalizing item {idx+1}/{total}: {item_id} ...", style='okblue')

    # ===== CORE FORMALIZATION =====
    
    def autoformalize_series(
        self,
        nl_proof: str = "",
        *,
        problem: str = "",
        raw_cot: str = "",
        graph_builder_retries: int = 3, 
        formalizer_retries: int = 3,
        prover_retries: int = 3,
        follow_dag: bool = True,
        previous_context: bool = True,
        supply_proof: bool = True,
        include_think_in_dag: bool = True) -> None:
        """
        Process a natural language proof through the complete formalization pipeline.
        
        This method performs the main proof formalization workflow:
        1. Builds a proof graph from the natural language proof
        2. Formalizes each proof step into Lean 4 code
        3. Attempts to automatically prove each formalized step
        
        Args:
            nl_proof: For task_profile='proof', the full natural-language theorem+proof text.
            problem/raw_cot: For task_profile='calc', the problem statement and model chain-of-thought.
            graph_builder_retries: Number of retries for proof graph generation
            formalizer_retries: Number of retries for each formalization step
            prover_retries: Number of retries for each proof generation step
            follow_dag: Whether to follow DAG structure or condition on all previous steps
            previous_context: Whether to provide dependency statements during formalization
            supply_proof: Whether to supply original proof text at each step for context
            include_think_in_dag: Whether DAG extraction may use <think>...</think> content.
        """
        if self.task_profile == "calc":
            if not problem or not raw_cot:
                raise ValueError("task_profile='calc' requires problem= and raw_cot=")
            self.nl_proof = f"Problem:\n{problem}\n\nRaw CoT:\n{raw_cot}"
        else:
            if not nl_proof:
                raise ValueError("task_profile='proof' requires nl_proof")
            self.nl_proof = nl_proof
        
        self._print_status("\nBuilding proof graph...", style='bold')

        #Step 1. Build proof graph
        self.proof_items, tries = build_proof_graph(
            model_manager=self.graph_model_manager,
            task_profile=self.task_profile,
            problem=problem,
            raw_cot=raw_cot,
            natural_language_proof=nl_proof,
            logs=self.llm_call_logs,
            follow_dag=follow_dag,
            max_retries=graph_builder_retries,
            id_schema_mode=self.id_schema_mode,
            validation_profile=self.validation_profile,
            allow_graph_rewrite_after=self.allow_graph_rewrite_after,
            include_think_in_dag=include_think_in_dag,
        )
        self._print_progress_summary(tries)

        for idx, item in enumerate(self.proof_items):
            self._print_item_progress(idx, len(self.proof_items), item.id)

            # Check if we should skip formalizing this node
            node_type = getattr(item, "node_type", None)
            needs_verification = int(getattr(item, "needs_verification", 0) or 0)
            is_condition_or_def = is_condition(item.id, node_type, mode=self.id_schema_mode) or (
                is_context(item.id, node_type, mode=self.id_schema_mode) and needs_verification != 1
            )

            if self.task_profile == "calc" and is_condition_or_def:
                # Skip formalizing context/condition nodes in calc mode; 
                # pass them as natural language to downstream claims instead.
                item.formalization = {
                    "lean_code": "",
                    "lean_pass": True,  # Pretend it passed so downstream doesn't complain
                    "error_msg": [],
                    "tries": 0,
                    "attempt_history": [],
                    "skipped": True
                }
                self._print_status(f"   \u2714 Skipped formalizing item {item.id} (passed as natural language).", style='okgreen')
                
                # Also skip solver
                item.solved_lemma = {}
                continue

            if self.task_profile == "calc":
                # For calc, we do not supply the problem or CoT to avoid the formalizer
                # trying to formalize the entire problem/solution inside a single node (like pc_1).
                context_to_supply = ""
            else:
                context_to_supply = self.nl_proof

            #Step 1. Formalize statement with Goedel-Formalizer
            formalization  = run_formalizer_prompt(
                item,
                lean_server=self.lean_server,
                all_items=self.proof_items,
                model_manager=self.formalize_model_manager,
                task_profile=self.task_profile,
                logs=self.llm_call_logs,
                max_retries=formalizer_retries,
                previous_context=previous_context,
                original_proof=context_to_supply if supply_proof else "",
                supply_proof=supply_proof,
            )
            item.formalization = formalization

            #Step 2. Solver with tactics
            solved_lemma = run_solver_prompt(
                item,
                lean_server=self.lean_server,
                model_manager=self.solver_model_manager,
                task_profile=self.task_profile,
                logs=self.llm_call_logs,
                max_retries=prover_retries,
                enable_ctx_solver=self.enable_ctx_solver,
            )
            item.solved_lemma = solved_lemma
        
            self._print_item_progress(idx, len(self.proof_items), item.id, completed=True)

    # ===== DATA ACCESS =====
    
    def get_llm_call_logs(self) -> List[Dict[str, Any]]:
        """
        Get logs of all LLM API calls made during processing.
        
        Returns:
            List of dictionaries containing call details (model, tokens, timing, etc.)
        """
        return self.llm_call_logs

    # ===== CODE GENERATION =====
    
    def get_lean_code(self) -> str:
        """
        Generate complete Lean 4 code from formalized proof items.
        
        Collects Lean code from each proof item, prioritizing solved lemmas over
        formalizations. Removes duplicate imports and combines all code blocks
        into a single, compilable Lean 4 file.
        
        Returns:
            Complete Lean 4 code with imports and all proof steps
            
        Raises:
            ValueError: If no proof items are available (run autoformalize_series first)
        """
        if self.proof_items is None:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        

        code_blocks = []
        
        for item in self.proof_items:

            # Skip if item is of type TheoremCondition (except if it has a conditions/solutions, in order words no "variable" in lean_code)
            if type(item).__name__ in ('TheoremCondition', 'Definition'):
                if hasattr(item, 'formalization'):
                    if "variable" in item.formalization.get('lean_code', ""):
                        continue
                else:
                    continue
                
            code = None
            # First check solved_lemma["lean_code"]
            if hasattr(item, 'solved_lemma') and item.solved_lemma not in (None, {}):
                if item.solved_lemma["lean_verify"]:
                    code = item.solved_lemma.get('lean_code')
            
            # If not found, check formalization["lean_code"]
            if not code and hasattr(item, 'formalization') and item.formalization not in (None, {}):
                code = item.formalization.get('lean_code')        
            if code:
                # Remove unwanted lines
                cleaned_code = remove_imports(code)
                if cleaned_code:
                    code_blocks.append(cleaned_code)
        
        # Add standard imports at the beginning
        header = """import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat Filter"""
        
        if code_blocks:
            return header + "\n\n" + "\n\n".join(code_blocks)
        else:
            return header

    # ===== VISUALIZATION =====
    
    def plot_dag(self, filepath: str = "proof_dag.png") -> None:
        """
        Create a static visualization of the proof dependency graph.
        
        Generates a PNG image showing the proof structure with nodes representing
        proof steps and edges showing dependencies between them.
        
        Args:
            filepath: Path where to save the PNG image
            
        Raises:
            ValueError: If no proof items are available
        """
        if not self.proof_items:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        
        graph, node_info = build_dag([item.model_dump() for item in self.proof_items])
        create_static_visualization(graph, node_info, filepath)
        
    def interactive_dag(self, filepath: str = "proof_dag.html") -> None:
        """
        Create an interactive HTML visualization of the proof dependency graph.
        
        Generates an interactive HTML file that allows exploring the proof structure
        with zoom, pan, and click interactions. Includes the original proof text.
        
        Args:
            filepath: Path where to save the HTML file
            
        Raises:
            ValueError: If no proof items are available
        """
        if not self.proof_items:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        
        graph, node_info = build_dag([item.model_dump() for item in self.proof_items])
        create_interactive_visualization(G = graph, node_info = node_info, proof_str = self.nl_proof, filename = filepath)

    # ===== ANALYSIS & METRICS =====
    
    def summary(self, verbose: bool = True, pass_at: int = 100000) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the proof formalization results.
        
        Calculates success rates for formalization and proof generation, along with
        performance metrics like token usage and processing time.
        
        Args:
            verbose: Whether to print summary information to console
            pass_at: Maximum number of attempts to consider for success calculation
            
        Returns:
            Dictionary containing:
                - form_total: Total number of formalizable steps
                - form_correct: Number of successfully formalized steps
                - solv_total: Total number of solvable steps
                - solv_correct: Number of successfully solved steps
                - form_acc: Formalization accuracy (0.0 to 1.0)
                - solv_acc: Proof success rate (0.0 to 1.0)
                - generated_tokens: Total tokens generated
                - total_calls: Total number of LLM calls made
                - total_time: Total processing time in seconds
        """
        # Print number of tc, l, ts
        if verbose:
            self._print_progress_summary(None)

        # Count formalization successes
        formalized_count = 0
        total_formalizable = 0
        formalizer_llm_calls = 0
        # Count solving successes
        solved_count = 0
        total_solvable = 0
        solver_llm_calls = 0

        for item in self.proof_items:
            # Check formalization status
            if hasattr(item, 'formalization') and item.formalization not in (None, {}):
                total_formalizable += 1
                formalizer_llm_calls += min(item.formalization["tries"], pass_at) if "tries" in item.formalization else 1
                if 'lean_pass' in item.formalization and item.formalization['lean_pass']:
                    if "tries" in item.formalization and item.formalization["tries"] <= pass_at:
                        formalized_count += 1

            
            # Check solving status
            if hasattr(item, 'solved_lemma') and item.solved_lemma not in (None, {}):
                total_solvable += 1
                solver_llm_calls += min(item.solved_lemma["tries"],  pass_at) if "tries" in item.solved_lemma else 1
                if 'lean_verify' in item.solved_lemma and item.solved_lemma['lean_verify']:
                    if item.solved_lemma["tries"] <= pass_at:
                        solved_count += 1

        if verbose:
            print(f"{formalized_count} steps successfully formalized out of {total_formalizable}")
            print(f"{solved_count} steps successfully solved out of {total_solvable}")

        #add gen_tokens, and total time, ignore scorer and negation prover
        new_tokens = 0
        new_llm_call_logs = []
        for call in self.llm_call_logs:
            if isinstance(call["generated_tokens"], int):
                if "**not to prove the given theorem/lemma" not in ''.join(d['content'] for d in call["messages"]):
                    new_tokens += call["generated_tokens"]
                    new_llm_call_logs.append(call)  # Append the call to the new list
        total_time = new_llm_call_logs[-1]['end_time']-new_llm_call_logs[0]['start_time']

        return {"form_total": total_formalizable,
                "form_correct": formalized_count,
                "solv_total": total_solvable,
                "solv_correct": solved_count,
                "form_acc": formalized_count/total_formalizable,
                "solv_acc": solved_count/total_solvable,
                "generated_tokens": new_tokens,
                "total_calls": 1 + formalizer_llm_calls + solver_llm_calls,
                "total_time": total_time}
    
    def elapsed_time(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze timing information for all LLM calls made during processing.
        
        Args:
            verbose: Whether to print timing breakdown to console
            
        Returns:
            List of dictionaries containing model name and elapsed time for each call
        """
        total_time = self.llm_call_logs[-1]['end_time']-self.llm_call_logs[0]['start_time']
        if verbose:
            print(f"Total time: {total_time:2f}")
            print()
        elapsed_times = []
        for current in self.llm_call_logs:
            model = current["model"]
            elapsed = current["end_time"] - current["start_time"]
            elapsed_times.append({"model": model, "elapsed_time":elapsed, "prove_negation": "**not to prove the given theorem/lemma" in ''.join(d['content'] for d in current["messages"])})
            if verbose:
                print(f"Model: {model:<30} | Time: {elapsed:.2f}")
        
        return elapsed_times

