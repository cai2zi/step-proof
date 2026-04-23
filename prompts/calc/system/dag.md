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

--------------------------------------------------
1. Contextual Text Extraction
--------------------------------------------------

- The "source_text" field must be an exact quote from either the problem text or the raw CoT text.
- Overlapping quotes are allowed.
- The "source_text" field must be a contiguous exact substring from the current Problem or current Raw CoT.
- Never copy "source_text" from the few-shot examples.
- Do NOT paraphrase, summarize, or synthesize "source_text".
- If a node combines several adjacent sentences from the Raw CoT, use the full contiguous span as "source_text".

--------------------------------------------------
2. Complete Coverage of Reasoning
--------------------------------------------------

- Include every meaningful deductive step in the CoT.
- Do NOT invent steps not supported by the text.
- Remove pure filler text ("let me think", "I continue", "this seems right") unless it introduces a real logical action.

--------------------------------------------------
3. Granular Decomposition
--------------------------------------------------

Default rule: one logical inference or one context-setting action per node.

Split when:
- a step uses a different premise set from adjacent steps.
- a step introduces a structurally distinct conclusion (e.g. case split, domain restriction, substitution result).

Merge when:
- multiple micro-operations are purely serial algebraic or arithmetic rewrites
  (e.g. multiply both sides → expand → collect terms → simplify → factor).
- they all depend on the same premises.
- a single natural statement can capture the input and output of the whole chain.

Key rule for basic transformations:
Collapse a chain of elementary rewrites (expand, collect, factor, cancel, substitute,
simplify) into ONE claim node whose statement reads
"[input expression] simplifies / transforms to [output expression]",
UNLESS an intermediate result is independently referenced by another node.

--------------------------------------------------
4. Dependency Management
--------------------------------------------------

- Each node must list only DIRECT dependencies in "dependencies".
- Do NOT include all ancestors.
- A node depends on another node only if that earlier node is a direct logical prerequisite.
- If a node depends only on the problem statement, dependencies may be empty.

--------------------------------------------------
5. No Error Correction
--------------------------------------------------

- Represent the CoT as written; do NOT fix mistakes or improve the proof.
- Only minor wording edits are allowed to make the "statement" self-contained.

--------------------------------------------------
6. Self-Contained Statement
--------------------------------------------------

- The "statement" field must be self-contained.
- Do NOT use vague references such as "this equation", "the above", "the previous result".
- Rewrite each statement so it can be understood independently, including relevant variables and assumptions.
- The later formalizer will only see the "statement" field.

--------------------------------------------------
7. Node Types
--------------------------------------------------

Use exactly one of the following values for "node_type":

1. "problem_condition" — facts directly extracted from the problem statement
   (variable domains, given equations, inequalities, initial constraints).
2. "context" — proof-local setup from the CoT
   (e.g. "assume x > 0", "let z = y^2", "consider case 2"). Not standalone derived conclusions.
3. "claim" — intermediate derived mathematical claims; the main reasoning steps.
4. "final_answer" — the final conclusion, solution set, or simplified expression.

Final-answer node rule:
- Use "final_answer" for the node that first states the final requested result.
- Do NOT create an extra final_answer node merely to restate a previous claim.
- If the last mathematical claim already computes or identifies the requested answer, make that node "final_answer" instead of "claim".
- Do NOT create a separate node only for rhetorical confirmation such as "checking the answer", "therefore", or "this matches the requirement", unless that sentence performs a necessary mathematical filtering step.

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

- "pc_1", "pc_2", ... for problem_condition
- "ctx_1", "ctx_2", ... for context
- "c_1", "c_2", ... for claim
- "fa_1", "fa_2", ... for final_answer

IDs must be unique.

--------------------------------------------------
10. needs_verification Rules
--------------------------------------------------

- Use 1 if this node is a good candidate for formal verification.
- Use 0 otherwise.

Typical defaults: problem_condition → 0, context → 0, claim → 1, final_answer → 1.
Use judgment: a pure formatting restatement may be 0; a mathematically meaningful conclusion should be 1.

--------------------------------------------------
Few-shot Example
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
    "source_text": "Multiply both sides by 6 to clear the denominators. Then the left side becomes 2y^4 - 6y^2 and the right side becomes 9y^2 + 27. Moving everything to one side gives 2y^4 - 15y^2 - 27 = 0.",
    "statement": "Multiplying (1/3)y^4 - y^2 = (3/2)y^2 + 9/2 by 6 and rearranging gives 2y^4 - 15y^2 - 27 = 0.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "ctx_1",
    "node_type": "context",
    "source_text": "Let z = y^2.",
    "statement": "Introduce the substitution z = y^2.",
    "dependencies": ["c_1"],
    "needs_verification": 0
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "Then we get 2z^2 - 15z - 27 = 0. Using the quadratic formula, the discriminant is 441, so the roots are 9 and -3/2.",
    "statement": "Under the substitution z = y^2, the equation 2y^4 - 15y^2 - 27 = 0 becomes 2z^2 - 15z - 27 = 0, which has roots z = 9 and z = -3/2 (discriminant 441).",
    "dependencies": ["c_1", "ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Since y^2 cannot be negative, z = -3/2 is invalid. Thus y^2 = 9, so y = ±3. Checking both values in the original equation shows that both work. Therefore the solution set is {3, -3}.",
    "statement": "Because z = y^2 >= 0, the root z = -3/2 is rejected; the valid root z = 9 gives the complete solution set {3, -3} for (1/3)y^4 - y^2 = (3/2)y^2 + 9/2.",
    "dependencies": ["pc_1", "ctx_1", "c_2"],
    "needs_verification": 1
  }
]

--------------------------------------------------
Few-shot Example 2
--------------------------------------------------

Problem:
A store sells notebooks for $2 each and pens for $1.50 each. If Mia buys 4 notebooks and 3 pens, how much does she pay in total?

Raw CoT:
The notebooks cost 4 * 2 = 8 dollars. The pens cost 3 * 1.50 = 4.50 dollars. Adding them gives 8 + 4.50 = 12.50 dollars.

Output:
[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "A store sells notebooks for $2 each and pens for $1.50 each. If Mia buys 4 notebooks and 3 pens, how much does she pay in total?",
    "statement": "Notebooks cost $2 each, pens cost $1.50 each, and Mia buys 4 notebooks and 3 pens.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "The notebooks cost 4 * 2 = 8 dollars.",
    "statement": "The cost of 4 notebooks at $2 each is $8.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "The pens cost 3 * 1.50 = 4.50 dollars.",
    "statement": "The cost of 3 pens at $1.50 each is $4.50.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Adding them gives 8 + 4.50 = 12.50 dollars.",
    "statement": "Mia pays $12.50 in total.",
    "dependencies": ["c_1", "c_2"],
    "needs_verification": 1
  }
]

--------------------------------------------------
Few-shot Example 3
--------------------------------------------------

Problem:
Solve 2/x + 5 >= 3/x. Express your answer in interval notation.

Raw CoT:
Move 2/x to the right to get 5 >= 1/x. Since x cannot be 0, consider x > 0 and x < 0. If x > 0, multiplying by x gives 5x >= 1, so x >= 1/5. If x < 0, multiplying by x reverses the inequality and gives 5x <= 1, which is true for all x < 0. Therefore the solution is (-infinity, 0) union [1/5, infinity).

Output:
[
  {
    "id": "pc_1",
    "node_type": "problem_condition",
    "source_text": "Solve 2/x + 5 >= 3/x. Express your answer in interval notation.",
    "statement": "Solve the inequality 2/x + 5 >= 3/x with x != 0, and express the solution in interval notation.",
    "dependencies": [],
    "needs_verification": 0
  },
  {
    "id": "c_1",
    "node_type": "claim",
    "source_text": "Move 2/x to the right to get 5 >= 1/x.",
    "statement": "The inequality 2/x + 5 >= 3/x is equivalent to 5 >= 1/x, with x != 0.",
    "dependencies": ["pc_1"],
    "needs_verification": 1
  },
  {
    "id": "ctx_1",
    "node_type": "context",
    "source_text": "Since x cannot be 0, consider x > 0 and x < 0.",
    "statement": "Split the analysis into the cases x > 0 and x < 0 because x cannot be 0.",
    "dependencies": ["pc_1", "c_1"],
    "needs_verification": 0
  },
  {
    "id": "c_2",
    "node_type": "claim",
    "source_text": "If x > 0, multiplying by x gives 5x >= 1, so x >= 1/5.",
    "statement": "In the case x > 0, the inequality 5 >= 1/x gives x >= 1/5.",
    "dependencies": ["c_1", "ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "c_3",
    "node_type": "claim",
    "source_text": "If x < 0, multiplying by x reverses the inequality and gives 5x <= 1, which is true for all x < 0.",
    "statement": "In the case x < 0, the inequality 5 >= 1/x is true for all x < 0.",
    "dependencies": ["c_1", "ctx_1"],
    "needs_verification": 1
  },
  {
    "id": "fa_1",
    "node_type": "final_answer",
    "source_text": "Therefore the solution is (-infinity, 0) union [1/5, infinity).",
    "statement": "The solution set of 2/x + 5 >= 3/x is (-infinity, 0) union [1/5, infinity).",
    "dependencies": ["c_2", "c_3"],
    "needs_verification": 1
  }
]
