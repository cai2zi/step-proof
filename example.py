import os

from dotenv import load_dotenv

from proofflow import LeanServer, LLMManager, ProofFlow

load_dotenv()
# Set up Lean server (local or remote)
# lean_server = LeanServer(api_url=os.getenv("LEAN_SERVER_URL"))  # Remote server
# OR
lean_server = LeanServer(project_path="/data/czx/mathlib4")  # Local project

# Configure vLLM (OpenAI-compatible) endpoints

GRAPH_BASE_URL = "http://127.0.0.1:8001/v1"
FORMALIZER_BASE_URL = "http://127.0.0.1:8002/v1"
PROVER_BASE_URL = "http://127.0.0.1:8003/v1"

# vLLM OpenAI-compatible API usually does not require a real key for localhost.
# Keep a non-empty placeholder string.
DUMMY_API_KEY = "dummy"

# Graph building model
graph_model = LLMManager(
    model_info={
        "api_key": DUMMY_API_KEY,
        "base_url": GRAPH_BASE_URL,
        "model": "qwen3.5-9b",
    },
)

# Formalization model
formalize_model = LLMManager(
    model_info={
        "api_key": DUMMY_API_KEY,
        "base_url": FORMALIZER_BASE_URL,
        "model": "goedel-formalizer-v2-8b",
    },
)

# Solver model
solver_model = LLMManager(
    model_info={
        "api_key": DUMMY_API_KEY,
        "base_url": PROVER_BASE_URL,
        "model": "goedel-prover-v2-8b",
    },
)

# Optional scoring model (disabled)
score_model = None
# Initialize ProofFlow
proof_flow = ProofFlow(
    lean_server=lean_server,
    graph_model_manager=graph_model,
    formalize_model_manager=formalize_model,
    solver_model_manager=solver_model,
    score_model_manager=score_model,
    verbose=True,
    task_profile="proof",
)
# Process a natural language proof
nl_proof = """
Theorem: For all real numbers x, y, if x² + y² = 1, then |x| ≤ 1.
Proof: Since x² ≥ 0 and y² ≥ 0, we have x² + y² ≥ x². 
Given that x² + y² = 1, we get 1 ≥ x², which means x² ≤ 1. 
Taking the square root of both sides, we obtain |x| ≤ 1.
"""

# Run formalization (proof task). For calculation-style DAGs use e.g.:
#   ProofFlow(..., task_profile="calc")
#   proof_flow.autoformalize_series(problem="...", raw_cot="...")
proof_flow.autoformalize_series(nl_proof)
# Get results
lean_code = proof_flow.get_lean_code()
print("Generated Lean 4 code:")
print(lean_code) 
#incorrectly formalized lemmas are not shown
# Get performance summary
summary = proof_flow.summary()
print(f"Formalization accuracy: {summary['form_acc']:.2%}")
print(f"Proof success rate: {summary['solv_acc']:.2%}")
# Generate visualizations
proof_flow.plot_dag("proof_dag.png")
proof_flow.interactive_dag("proof_dag.html")
print("Visualizations saved as proof_dag.png and proof_dag.html")
