FAIL_RATE_PROMPT = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{instruction}
""".strip()


# 下面这版就是 **“最大程度对齐原始 ProofFlow formalizer、同时适配你当前 block schema”** 的最终版 prompt。

# 先说明两点映射规则：

# * 你的代码在调用这个 prompt 前，先把当前 block 映射成：

#   * `lemma_header`：

#     * 若 `block.node_type == "final_answer"`，则用 `theorem {block.id}`
#     * 否则用 `lemma {block.id}`
#   * `statement`：直接用 `block.statement`
#   * `dependencies`：直接用 `block.dependencies`
#   * `dependency_lean_code`：把依赖节点对应的 Lean 代码拼接进去
# * 这版 prompt **不再显式引入 `problem` 和 `source_text`**，并且恢复原始 ProofFlow formalizer 的：

#   * `lemma_header`
#   * `statement`
#   * `dependencies`
#   * `Lean skeleton with placeholders`
#   * `prior Lean code`
#     这几项核心交互结构。原始 prompt 就是按这个接口设计的，并要求模型只替换 skeleton 中的 hypothesis 与 goal 部分。




Formalization_PROMPT = """

# System Prompt — Lean Lemma Auto-Formalization

You are a thinking model specialized in turning natural-language mathematical proof steps into Lean 4 code.

You will receive:

1. A lemma name (`lemma_header`, e.g. `lemma l3` or `theorem ts_1`)
2. A natural-language statement for the current proof step (`statement`)
3. The names of dependencies (`dependencies`)
4. A Lean code skeleton with placeholders
5. Full Lean 4 code of prior steps (`dependency_lean_code`)

Your job is to emit exactly one Lean 4 code block that compiles after replacing the placeholders, while not reproducing any prior lemmas or defs. Treat prior steps only as hypotheses.

---

## Hard Rules (highest priority)

0. Single lemma only
- Output must contain exactly one lemma/theorem definition (the given `lemma_header`).
- The fenced block must contain exactly one of: `lemma ` or `theorem `.
- The block must contain exactly one occurrence of `:= by` and exactly one occurrence of `sorry`.

1. Provide exactly one fenced Lean block
- Think carefully before giving the final answer.
- At the end, write exactly one fenced Lean block and nothing else.

2. Header is fixed
Keep exactly this header in this order:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter
````

3. Proof body format

* The proof body must be exactly:
  := by
  sorry

4. Lemma name unchanged

* Keep the `lemma_header` name and structure exactly.
* Only fill in parameters/hypotheses and the goal.

5. Prohibited content inside the block

* Do not generate any additional `lemma`, `theorem`, `def`, `example`, `instance`, `axiom`, `structure`, or extra `import` / `set_option` / `open` sections beyond the required header.

---

## Using Dependencies

Often you are given the full Lean code of earlier steps (“dependencies”).
These prior lemmas contain relevant Lean 4 information that you may use when formalizing the current step, including:

* type declarations
* formalized proposition types of the previous lemmas
* constants and structures appearing in those lemmas

What you must do with dependencies:

1. Include every dependency as a named hypothesis in the current lemma.

   * Use the dependency lemma’s name as the hypothesis name.
   * Its hypothesis type is the proposition type of that prior lemma, adapted to match the variables in the current lemma.
   * Include all dependencies as hypotheses even if you think they are not relevant or will not be used.

2. You do not need to reproduce dependency code verbatim.

   * Prior code is for reference only.
   * If needed, adapt dependency statements so they fit the current lemma’s parameters and context.
   * Instantiate with current variables when possible, or generalize with ∀ if necessary.
   * Minor modifications such as variable renaming or type adaptation are allowed if needed for consistency.

3. Preferred behavior

   * If the current lemma declares matching variables (same names/types), instantiate the dependency with those variables and do not introduce unnecessary ∀-quantification.

4. Extra hypotheses are allowed

   * You may add new hypotheses such as type declarations or variable declarations if they are needed to make the current lemma well-formed.
   * Always place these before dependency hypotheses in the parameter list.

5. Do not duplicate a dependency that you have already added.

6. If no dependencies are provided, skip this section and proceed normally.

---

## Parameter / Hypothesis Ordering

When you fill the skeleton parameters, place items in this order:

1. Type / implicit parameters first
2. Explicit variable declarations next
3. Dependency hypotheses (one per prior lemma, named by lemma name; adapted as needed)

Then place:

* a colon `:`
* the formalized goal statement
* followed by `:= by`
* followed by `sorry`

---

## Goal Formalization

* Replace `[place correct hypothesis here]` and `[place goal here]` with the precise Lean proposition matching the natural-language statement.
* Preserve the meaning of the natural-language statement as faithfully as possible.
* Use standard Lean 4 / Mathlib 4 syntax.
* Do not improve the mathematics; only formalize the given step.
* If the statement is underspecified, infer only the minimal missing variable/type information needed to make the current block well-formed.
* Do not add extra mathematical claims not supported by the statement or dependencies.

---

## Skeleton You Must Fill

You will be given content like:

\"\"\"
Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {statement}
The dependencies are: {dependencies}

This is the lean code skeleton you need to use:

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

\"\"\"

The natural-language statement may already be self-contained and may mention prior steps only indirectly.
Use the dependency names and prior Lean code to reconstruct the correct hypotheses.

Your output must replace:

* `[place correct hypothesis here]` with the ordered parameter/hypothesis list
* `[place goal here]` with the formalized goal

The final fenced block must consist of:

* the fixed header
* exactly one `lemma_header` definition
* and nothing else

---

## Few-Shot Examples

### ✅ Example 1 — Preserve dependency names and ordering

Input:

* Lemma Name: `theorem ts_1`
* Natural Language Statement:
  "Assuming a ≥ 0 and b ≤ 0 [tc_1] and abs (a + b) = abs a + abs b [l1], we conclude that |a + b| ≤ |a| + |b| [ts_1]."
* Dependencies: `["tc_1", "l1"]`
* Lean4 code of prior steps: omitted here

Correct Output:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

theorem ts_1
  (a b : ℝ)
  (tc_1 : a ≥ 0 ∧ b ≤ 0)
  (l1 : abs (a + b) = abs a + abs b) :
  |a + b| ≤ |a| + |b| := by
  sorry
```

Wrong behavior:

* renaming `ts_1` to something else
* renaming `tc_1` or `l1` to `h1`, `h2`
* adding extra theorem definitions

---

### ✅ Example 2 — Add missing variable types

Input:

* Lemma Name: `lemma l3`
* Natural Language Statement:
  "Assuming 2 * d = 6 [l2], we conclude that d = 3 [l3]."
* Dependencies: `["l2"]`
* Lean4 code of prior step `l2`: omitted here

Correct Output:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

lemma l3
  (d : ℝ)
  (l2 : 2 * d = 6) :
  d = 3 := by
  sorry
```

Why correct:

* The missing variable `(d : ℝ)` is added.
* The dependency name `l2` is preserved.

---

### ✅ Example 3 — Use dependency proposition type directly

Input:

* Lemma Name: `lemma c_4`
* Natural Language Statement:
  "Under the substitution z = y^2 [ctx_1] and the equation 2y^4 - 15y^2 - 27 = 0 [c_3], we conclude that 2z^2 - 15z - 27 = 0 [c_4]."
* Dependencies: `["c_3", "ctx_1"]`
* Lean4 code of prior steps:

  * `lemma c_3 (y : ℝ) : 2 * y^4 - 15 * y^2 - 27 = 0 := by sorry`
  * `lemma ctx_1 (y z : ℝ) : z = y^2 := by sorry`

Correct Output:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

lemma c_4
  (y z : ℝ)
  (c_3 : 2 * y^4 - 15 * y^2 - 27 = 0)
  (ctx_1 : z = y^2) :
  2 * z^2 - 15 * z - 27 = 0 := by
  sorry
```

---

## Now process the following input

Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {statement}
The dependencies are: {dependencies}

This is the lean code skeleton you need to use:

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

Here is the full Lean 4 code of prior dependency steps:

{dependency_lean_code}

```

"""


DAG_PROMPT="""
You are an expert at analyzing mathematical chain-of-thought reasoning and converting it into a structured proof DAG for later Lean4-oriented verification.

You are given:
1. A natural language math problem.
2. A raw chain-of-thought (CoT) solution trace.

Your task is to generate a JSON ARRAY of DAG nodes that faithfully represents the logical structure of the given CoT.

Important:
- Output ONLY a valid JSON array.
- Do NOT output markdown.
- Do NOT output explanations.
- Do NOT wrap the result in {"response": ...}.
- Do NOT include any outer id field.
- Each array element must be one node object.


Downstream interpretation:
- problem_condition nodes are foundational facts extracted from the problem and are usually formalized only.
- context nodes are proof-local setup steps and are usually formalized only.
- claim nodes are intermediate derived facts and are usually formalized and then proved.
- final_answer nodes are final conclusions and are usually formalized and then proved.

Your graph must satisfy the following principles.

--------------------------------------------------
1. Contextual Text Extraction
--------------------------------------------------

- The "source_text" field must be an exact quote from either:
  - the problem text, or
  - the raw CoT text
- Overlapping quotes are allowed.
- The goal is to preserve traceability, not to partition the text.

--------------------------------------------------
2. Complete Coverage of Reasoning
--------------------------------------------------

- Include every meaningful deductive step in the CoT.
- Do NOT invent steps that are not supported by the text.
- Do NOT skip "obvious" steps if they are part of the author's reasoning.
- Remove pure filler text such as:
  - "let me think"
  - "I continue"
  - "this seems right"
  - "let me check again"
  unless it introduces a real logical action such as defining a variable, splitting into cases, or checking candidates.

--------------------------------------------------
3. Granular Decomposition (with merge nuance)
--------------------------------------------------

Default rule:
- one logical inference or one context-setting action per node.

Split when:
- a sentence performs heterogeneous reasoning,
- a sentence uses different premise sets,
- a sentence contains both a derived fact and a later consequence.

Merge when:
- multiple micro-operations are purely local calculations or rewrites
- a single short tactic or local algebraic simplification would naturally handle them together

Heuristic:
- If two actions rely on different lemmas/assumptions, split them.
- If they are serial simplifications within one formula and would be discharged by one short tactic, merge them.
- If a sentence says "X, which gives Y, so Z":
  - split into X→Y and Y→Z
  - unless Y is only a trivial rewrite that the same verification step would subsume

--------------------------------------------------
4. Dependency Management
--------------------------------------------------

- Each node must list only DIRECT dependencies in "dependencies".
- Do NOT include all ancestors.
- A node should depend on another node if and only if that earlier node is a direct logical prerequisite.
- If a node depends only on the problem statement and no earlier derived node, dependencies may be empty.
- If the CoT says something vague like "by the above", include all reasonable prior nodes that directly support the step.

--------------------------------------------------
5. No Error Correction
--------------------------------------------------

- Represent the CoT as written.
- Do NOT fix mistakes.
- Do NOT improve the proof.
- Do NOT replace weak reasoning with stronger reasoning.
- Only minor wording edits are allowed to make the "statement" field self-contained and precise.

--------------------------------------------------
6. Self-Contained Statement
--------------------------------------------------

- The "statement" field must be self-contained.
- Do NOT use vague references such as:
  - "this equation"
  - "the above"
  - "that condition"
  - "the previous result"
- Rewrite each statement so that it can be understood independently.
- Include relevant variable/domain/assumption information when needed.
- The later formalizer will only see the "statement" field, so it must be self-contained.

--------------------------------------------------
7. Node Types
--------------------------------------------------

Use exactly one of the following values for "node_type":

1. "problem_condition"
   - facts directly extracted from the problem statement
   - variable domains, given equations, inequalities, divisibility conditions, initial constraints

2. "context"
   - proof-local setup or structural actions from the CoT
   - examples:
     - assume x > 0
     - consider case 2
     - let z = y^2
     - now check the candidates
   - these are not standalone derived conclusions

3. "claim"
   - intermediate derived mathematical claims from the CoT
   - these are the main reasoning steps

4. "final_answer"
   - the final conclusion, final solution set, or final simplified expression

--------------------------------------------------
8. Required Output Keys
--------------------------------------------------

Each node object must contain exactly these keys:

- "id"
- "node_type"
- "source_text"
- "statement"
- "dependencies"
- "needs_verification"

Do NOT include any other keys.

--------------------------------------------------
9. ID Rules
--------------------------------------------------

Use the following prefixes:
- "pc_1", "pc_2", ... for problem_condition
- "ctx_1", "ctx_2", ... for context
- "c_1", "c_2", ... for claim
- "fa_1", "fa_2", ... for final_answer

IDs must be unique.

--------------------------------------------------
10. needs_verification Rules
--------------------------------------------------

- Use 1 if this node is a good candidate for later formal verification.
- Use 0 otherwise.

Typical defaults:
- problem_condition: usually 0
- context: usually 0
- claim: usually 1
- final_answer: usually 1

But use judgment:
- a pure formatting-only restatement may be 0
- a mathematically meaningful final conclusion should be 1

--------------------------------------------------
11. Output Format
--------------------------------------------------

Output a JSON array like this:

[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "...",
    "statement": "...",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "ctx_1",
    "node_type": "context",
    "source_text": "...",
    "statement": "...",
    "dependencies": ["pc_1"],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "...",
    "statement": "...",
    "dependencies": ["ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "...",
    "statement": "...",
    "dependencies": ["c_1"],
    "needs_verification": 1
  }
]

--------------------------------------------------
Few-shot Example 1
--------------------------------------------------

Problem:
If A ⊆ B and B ⊆ C, prove that A ⊆ C.

Raw CoT:
Let x be an element of A. Since A is a subset of B, x is also in B. Since B is a subset of C, x is in C. Therefore A is a subset of C.

Output:
[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "A ⊆ B",
    "statement": "A is a subset of B.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "pc_2",
    "node_type": "problem_condition",
    "source_text": "B ⊆ C",
    "statement": "B is a subset of C.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "ctx_1",
    "node_type": "context",
    "source_text": "Let x be an element of A.",
    "statement": "Assume x is an element of A.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "Since A is a subset of B, x is also in B.",
    "statement": "From x ∈ A and A ⊆ B, derive x ∈ B.",
    "dependencies": ["pc_1", "ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "Since B is a subset of C, x is in C.",
    "statement": "From x ∈ B and B ⊆ C, derive x ∈ C.",
    "dependencies": ["pc_2", "c_1"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Therefore A is a subset of C.",
    "statement": "For any x, if x ∈ A then x ∈ C; therefore A ⊆ C.",
    "dependencies": ["ctx_1", "c_2"],
    "needs_verification": 1
  }
]

--------------------------------------------------
Few-shot Example 2
--------------------------------------------------

Problem:
Solve the equation (1/3)y^4 - y^2 = (3/2)y^2 + 9/2.

Raw CoT:
Multiply both sides by 6 to clear the denominators. Then the left side becomes 2y^4 - 6y^2 and the right side becomes 9y^2 + 27. Moving everything to one side gives 2y^4 - 15y^2 - 27 = 0. Let z = y^2. Then we get 2z^2 - 15z - 27 = 0. Using the quadratic formula, the discriminant is 441, so the roots are 9 and -3/2. Since y^2 cannot be negative, z = -3/2 is invalid. Thus y^2 = 9, so y = ±3. Checking both values in the original equation shows that both work. Therefore the solution set is {3, -3}.

Output:
[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "Solve the equation (1/3)y^4 - y^2 = (3/2)y^2 + 9/2.",
    "statement": "The given equation is (1/3)y^4 - y^2 = (3/2)y^2 + 9/2.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "Multiply both sides by 6 to clear the denominators.",
    "statement": "Multiplying both sides of (1/3)y^4 - y^2 = (3/2)y^2 + 9/2 by 6 is a valid transformation.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "Then the left side becomes 2y^4 - 6y^2 and the right side becomes 9y^2 + 27.",
    "statement": "After multiplying the equation by 6, the left side becomes 2y^4 - 6y^2 and the right side becomes 9y^2 + 27.",
    "dependencies": ["c_1"],
    "needs_verification": 1
  },
  {
    "id": "c_3",
    "node_type": "claim",
    "source_text": "Moving everything to one side gives 2y^4 - 15y^2 - 27 = 0.",
    "statement": "Rearranging 2y^4 - 6y^2 = 9y^2 + 27 gives 2y^4 - 15y^2 - 27 = 0.",
    "dependencies": ["c_2"],
    "needs_verification": 1
  },
  {
    "id": "ctx_1",
    "node_type": "context",
    "source_text": "Let z = y^2.",
    "statement": "Introduce the substitution z = y^2.",
    "dependencies": ["c_3"],
    "needs_verification": 0
  },
  {
    "id": "c_4",
    "node_type": "claim",
    "source_text": "Then we get 2z^2 - 15z - 27 = 0.",
    "statement": "Under the substitution z = y^2, the equation 2y^4 - 15y^2 - 27 = 0 becomes 2z^2 - 15z - 27 = 0.",
    "dependencies": ["c_3", "ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "c_5",
    "node_type": "claim",
    "source_text": "Using the quadratic formula, the discriminant is 441, so the roots are 9 and -3/2.",
    "statement": "For the quadratic equation 2z^2 - 15z - 27 = 0, the discriminant is 441 and its roots are z = 9 and z = -3/2.",
    "dependencies": ["c_4"],
    "needs_verification": 1
  },
  {
    "id": "c_6",
    "node_type": "claim",
    "source_text": "Since y^2 cannot be negative, z = -3/2 is invalid.",
    "statement": "Because z = y^2 and y^2 cannot be negative, the root z = -3/2 is invalid.",
    "dependencies": ["ctx_1", "c_5"],
    "needs_verification": 1
  },
  {
    "id": "c_7",
    "node_type": "claim",
    "source_text": "Thus y^2 = 9, so y = ±3.",
    "statement": "From the valid root z = 9 and the substitution z = y^2, derive y^2 = 9 and hence y = ±3.",
    "dependencies": ["ctx_1", "c_5", "c_6"],
    "needs_verification": 1
  },
  {
    "id": "ctx_2",
    "node_type": "context",
    "source_text": "Checking both values in the original equation shows that both work.",
    "statement": "Check the candidate values y = 3 and y = -3 in the original equation.",
    "dependencies": ["pc_1", "c_7"],
    "needs_verification": 0
  },
  {
    "id": "c_8",
    "node_type": "claim",
    "source_text": "Checking both values in the original equation shows that both work.",
    "statement": "Both candidate values y = 3 and y = -3 satisfy the original equation (1/3)y^4 - y^2 = (3/2)y^2 + 9/2.",
    "dependencies": ["pc_1", "c_7", "ctx_2"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Therefore the solution set is {3, -3}.",
    "statement": "The complete solution set is {3, -3}.",
    "dependencies": ["c_8"],
    "needs_verification": 1
  }
]

--------------------------------------------------
Few-shot Example 3
--------------------------------------------------

Problem:
Let x be a real number. If we triple it, add 12, subtract 6x from the resulting expression, and then divide the difference by 3, what is the final expression in simplest form?

Raw CoT:
Tripling x gives 3x. Adding 12 to 3x gives 3x + 12. Subtracting 6x from 3x + 12 gives (3x + 12) - 6x. Simplifying gives -3x + 12. Dividing by 3 gives (-3x + 12)/3. Simplifying again gives -x + 4. Therefore the final expression is -x + 4.

Output:
[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "Let x be a real number.",
    "statement": "x is a real number.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "Tripling x gives 3x.",
    "statement": "Tripling x gives 3x.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "Adding 12 to 3x gives 3x + 12.",
    "statement": "Adding 12 to 3x gives 3x + 12.",
    "dependencies": ["c_1"],
    "needs_verification": 1
  },
  {
    "id": "c_3",
    "node_type": "claim",
    "source_text": "Subtracting 6x from 3x + 12 gives (3x + 12) - 6x.",
    "statement": "Subtracting 6x from 3x + 12 gives (3x + 12) - 6x.",
    "dependencies": ["c_2"],
    "needs_verification": 1
  },
  {
    "id": "c_4",
    "node_type": "claim",
    "source_text": "Simplifying gives -3x + 12.",
    "statement": "Simplifying (3x + 12) - 6x gives -3x + 12.",
    "dependencies": ["c_3"],
    "needs_verification": 1
  },
  {
    "id": "c_5",
    "node_type": "claim",
    "source_text": "Dividing by 3 gives (-3x + 12)/3.",
    "statement": "Dividing -3x + 12 by 3 gives (-3x + 12)/3.",
    "dependencies": ["c_4"],
    "needs_verification": 1
  },
  {
    "id": "c_6",
    "node_type": "claim",
    "source_text": "Simplifying again gives -x + 4.",
    "statement": "Simplifying (-3x + 12)/3 gives -x + 4.",
    "dependencies": ["c_5"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Therefore the final expression is -x + 4.",
    "statement": "The final expression in simplest form is -x + 4.",
    "dependencies": ["c_6"],
    "needs_verification": 1
  }
]

Now process the following input.

Problem:
{PROBLEM}

Raw CoT:
{RAW_COT}
"""