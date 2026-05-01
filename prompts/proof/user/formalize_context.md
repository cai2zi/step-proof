Please autoformalize the following natural language context/condition in Lean 4.
Use the following name: {node_id}

The natural language statement is: {statement}

These the lean code skeleton you need to use (please make needed changes and fill ????):

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

variable [place correct hypothesis here]
```

Do not produce a theorem or a proof. Only provide the Lean 4 code.
Warning: this is a context/condition node. For this problem make use of "variable" and follow the following examples.

Name: tc\_1; Statement: Let \$(a\_n)\$ be a sequence of positive real numbers.
Lean 4 formalization:

```lean4
variable (a : ℕ → ℝ)
(tc_1 : ∀ n, 0 < a n)
```

Name: tc\_2; Statement: Let \$A\$ be a \$2 × 2\$ real matrix with eigenvalues \$\lambda\_1 = 3\$ and \$\lambda\_2 = -2\$.
Lean 4 formalization:

```lean4
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
(tc_2 : ∃ v1 v2 : Fin 2 → ℝ, v1 ≠ 0 ∧ v2 ≠ 0 ∧ A.vecMul v1 = 3 • v1 ∧ A.vecMul v2 = -2 • v2)
```
{dependency_context_block}{original_proof_block}
