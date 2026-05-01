import os
from dotenv import load_dotenv
from proofflow.lean_check import LeanServer, process_lean_string

load_dotenv()
MATHLIB_PATH = os.getenv("MATHLIB_PROJECT_PATH", "/data/czx/mathlib4")

# Edit this string to test your Lean 4 code
lean_code = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

variable (x : ℝ)
variable (pc_1 : x > 0)
"""

def main():
    print("Initializing LeanServer...")
    server = LeanServer(project_path=MATHLIB_PATH)
    
    print("\nTesting code:")
    print("-" * 40)
    print(lean_code)
    print("-" * 40)
    
    processed_code = process_lean_string(lean_code)
    pass_flag, verify_flag, error_msg = server.check_lean_string(processed_code)
    
    print(f"\nPass (Syntax): {pass_flag}")
    print(f"Verify (Proof): {verify_flag}")
    if error_msg:
        print("Errors:")
        for err in error_msg:
            print(err)

if __name__ == "__main__":
    main()
