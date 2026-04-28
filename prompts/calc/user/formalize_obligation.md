Please formalize the following closed proof obligation in Lean 4.

Use the following theorem name: {lemma_header}

The obligation is:
{informal_statement_content}

This is the Lean code skeleton you must fill:

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
