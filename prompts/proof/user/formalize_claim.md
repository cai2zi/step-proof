Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {statement}
The dependencies are: {dependencies}

This is the  lean code skeleton you need to use:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

{lemma_header}
[place correct hypothesis here] :
[place goal here] := by
sorry
```

Important: **Please write only one lemma or theorem**!!
{dependency_context_block}{original_proof_block}
