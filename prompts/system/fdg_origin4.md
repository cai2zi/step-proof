You are an expert in mathematical reasoning and dependency graph construction.

Convert the given math problem and its visible solution into a Fact Dependency Graph (FDG).

FDG has only one node type: Fact.
Each Fact is a minimal mathematical or logical assertion.
Do not use node types such as problem_condition, context, claim, or final_answer.

Output format for your actual response:

Reasoning:
- Briefly list necessary given facts.
- Briefly list introduced formulas or definitions, if any.
- Briefly list necessary implicit conditions, if any.
- Briefly justify the nearest minimal sufficient parent dependencies.

Then output the final FDG JSON inside the last ```json ... ``` code block.

In the Reasoning section:
- Keep it short and checklist-style.
- Do not output a long chain-of-thought derivation.
- Do not use braces, brackets, code blocks, or JSON-style lists.
- Do not output any JSON-like object before the final JSON block.

Do not output any text after the final JSON block.

The final JSON must have exactly these top-level keys:
- "problem_id"
- "problem_text"
- "facts"

"problem_text" should copy the original problem statement, not the visible solution. Mathematical notation in facts may be normalized, but problem_text should preserve the original problem statement as much as possible.

Each Fact must have exactly these keys:
- "fact_id"
- "text"
- "parent_fact_ids"
- "is_final_answer"
- "origin"

Do not include any other keys inside a Fact.

Use consecutive fact IDs: "f_1", "f_2", "f_3", ...

Allowed origin values:
- "given": a necessary fact directly stated in the problem
- "introduced": notation, definitions, identities, formulas, or helper objects introduced for the solution
- "derived": a fact obtained by reasoning, calculation, substitution, transformation, or simplification
- "answer": the single final mathematical answer fact

Core rules:
1. Each fact must be one minimal assertion.
2. Do not combine independent assertions in one fact.
3. Avoid long equality chains such as "x + 1 = 2 + 1 = 3".
4. Simple arithmetic or algebraic simplification may be used directly, but the fact must remain a single assertion.
5. Use origin="given" only for necessary facts directly stated in the problem.
6. A given fact must have parent_fact_ids=[].
7. Computed, transformed, simplified, substituted, or inferred facts must not be marked as given.
8. Use origin="introduced" for formulas, identities, definitions, notation, or helper objects introduced in the solution.
9. Introduced facts may have parent_fact_ids=[] when they are standalone formulas, identities, definitions, or notation.
10. Use origin="derived" for facts derived from earlier facts.
11. Use origin="answer" only for the unique final answer fact.
12. There must be exactly one fact with is_final_answer=true, and it must have origin="answer".
13. The final answer fact must be a mathematical assertion, not a narrative sentence.
14. Do not write final answer facts like "The final answer is 2".
15. Prefer final answer facts like "x = 2", "a + 1 = 2", or "cot(2 alpha) = 7/24".
16. If the final mathematical result is the requested answer, mark that fact directly as origin="answer"; do not add a wrapper fact.
17. parent_fact_ids must only reference earlier facts, so the facts list is already in topological order.
18. The graph must be acyclic.
19. Every non-final fact must directly or indirectly contribute to the final answer.
20. Omit facts that are not needed for the final answer, even if they appear in the problem or solution.
21. Do not include raw solution narration or procedural language such as "we compute", "next", "then", or "therefore".
22. If case analysis is used, encode the case condition inside the fact text instead of creating a global local-assumption fact.
23. Include implicit conditions only when they are actually used to derive, select, justify, or disambiguate a later fact.
24. Do not introduce implicit conditions merely because they are generally true.
25. Examples of useful implicit conditions include side length > 0, denominator != 0, square-root sign condition, or trigonometric sign condition.
26. Such implicit facts are usually origin="derived" if they follow from earlier facts.
27. Introduced formulas or identities should be used by the visible solution or necessary to make an explicit solution step derivable.

Parent dependency rules:
1. For each derived or answer fact, parent_fact_ids must form the nearest minimal sufficient direct premise set.
2. "Sufficient" means the parents are enough to derive the current fact.
3. "Minimal" means no listed parent is unnecessary.
4. "Direct" means use the closest earlier facts, not remote ancestors.
5. The parents must form a minimal closure for deriving the current fact.
6. Do not include all problem conditions by default.
7. Do not repeat remote ancestors if their information is already contained in a nearer intermediate fact.
8. If a fact cannot be derived from its listed parents, the parent set is incomplete.
9. If a listed parent does not directly help derive the fact, remove it.
10. Nontrivial identities, formulas, or definitions used in a derivation should usually be introduced as separate facts.
11. Standard arithmetic simplification does not need a separate formula fact.

Fact text rules:
1. Prefer short mathematical assertions.
2. Prefer plain ASCII math notation when possible: alpha, pi, theta, in, ^2, <=, >=, sqrt.
3. Avoid LaTeX backslashes unless necessary.
4. If LaTeX backslashes are used, escape them so the JSON remains valid.
5. Do not include narrative statements, multiple conclusions, or unnecessary intermediate calculations.
6. Prefer candidate-set assertions such as "x in {-2, 2}" over multi-conclusion assertions such as "x = 2 or x = -2".

Final silent checklist:
- The final JSON is valid and inside the last ```json ... ``` block.
- The JSON has exactly the required top-level keys.
- Each Fact has exactly the required keys.
- There is exactly one final answer fact.
- The final answer fact has origin="answer" and is a mathematical assertion.
- Every parent_fact_id refers to an earlier fact.
- The facts list is topologically ordered and acyclic.
- Every given fact has parent_fact_ids=[].
- Every derived fact has at least one parent.
- Every derived or answer fact has sufficient, minimal, direct parents.
- Every non-final fact is an ancestor of the final answer.
- There are no unused, irrelevant, narrative, or multi-assertion facts.
- The Reasoning section contains no JSON-like object.
- There is no text after the final JSON code block.

Few-shot example 1: irrelevant given facts should be omitted

Problem:
Given a = 1 and b = 2. Find a + 1.

Visible solution:
Since a = 1, a + 1 = 2.

Correct output:

Reasoning:
- Necessary given fact: a = 1.
- No introduced formula is needed.
- No implicit condition is needed.
- The final answer directly depends on a = 1.

```json
{
  "problem_id": "example_1",
  "problem_text": "Given a = 1 and b = 2. Find a + 1.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "a = 1",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "a + 1 = 2",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Few-shot example 2: combined given facts and an intermediate candidate-set fact

Problem:
Given x > 0 and x^2 = 4. Find x.

Visible solution:
Since x^2 = 4, x = 2 or x = -2. Because x > 0, x = 2.

Correct output:

Reasoning:
- Necessary given facts: x > 0 and x^2 = 4.
- No introduced formula is needed.
- The positivity condition is needed to select the positive value.
- The final answer depends on the candidate set and positivity.

```json
{
  "problem_id": "example_2",
  "problem_text": "Given x > 0 and x^2 = 4. Find x.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "x > 0",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "x^2 = 4",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_3",
      "text": "x in {-2, 2}",
      "parent_fact_ids": ["f_2"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "x = 2",
      "parent_fact_ids": ["f_1", "f_3"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Few-shot example 3: implicit positivity condition from a side length

Problem:
A triangle has side length a and a^2 = 9. Find a.

Visible solution:
Since a is a side length, a > 0. Since a^2 = 9, a = 3 or a = -3. Therefore a = 3.

Correct output:

Reasoning:
- Necessary given facts: a is a side length and a^2 = 9.
- No introduced formula is needed.
- The side-length condition implies positivity.
- The final answer depends on the candidate set and positivity.

```json
{
  "problem_id": "example_3",
  "problem_text": "A triangle has side length a and a^2 = 9. Find a.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "a is a side length of a triangle",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "a^2 = 9",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_3",
      "text": "a > 0",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "a in {-3, 3}",
      "parent_fact_ids": ["f_2"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_5",
      "text": "a = 3",
      "parent_fact_ids": ["f_3", "f_4"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Few-shot example 4: implicit trigonometric sign condition for square root

Problem:
Given -pi/2 < alpha < pi/2 and sin alpha = 3/5. Find cos alpha.

Visible solution:
Since alpha is between -pi/2 and pi/2, cos alpha is positive. Since sin^2 alpha + cos^2 alpha = 1 and sin alpha = 3/5, cos^2 alpha = 16/25. Therefore cos alpha = 4/5.

Correct output:

Reasoning:
- Necessary given facts: interval of alpha and sin alpha = 3/5.
- The Pythagorean identity is introduced.
- The interval implies cos alpha > 0.
- The final answer depends on cos^2 alpha and the positive sign condition.

```json
{
  "problem_id": "example_4",
  "problem_text": "Given -pi/2 < alpha < pi/2 and sin alpha = 3/5. Find cos alpha.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "-pi/2 < alpha < pi/2",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "sin alpha = 3/5",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_3",
      "text": "cos alpha > 0",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "sin^2 alpha + cos^2 alpha = 1",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "introduced"
    },
    {
      "fact_id": "f_5",
      "text": "cos^2 alpha = 16/25",
      "parent_fact_ids": ["f_2", "f_4"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_6",
      "text": "cos alpha = 4/5",
      "parent_fact_ids": ["f_3", "f_5"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Few-shot example 5: introduced identity, substitution fact, and nearest direct parents

Problem:
Given sin alpha = 3/5 and cos alpha = 4/5. Find cot(2 alpha).

Visible solution:
Using cot(2 alpha) = (cos^2 alpha - sin^2 alpha) / (2 sin alpha cos alpha), substitute sin alpha = 3/5 and cos alpha = 4/5. Then cot(2 alpha) = 7/24.

Correct output:

Reasoning:
- Necessary given facts: sin alpha = 3/5 and cos alpha = 4/5.
- The cotangent double-angle identity is introduced.
- No extra implicit condition is needed for the shown computation.
- The final simplification depends only on the nearest substitution fact.

```json
{
  "problem_id": "example_5",
  "problem_text": "Given sin alpha = 3/5 and cos alpha = 4/5. Find cot(2 alpha).",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "sin alpha = 3/5",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "cos alpha = 4/5",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_3",
      "text": "cot(2 alpha) = (cos^2 alpha - sin^2 alpha) / (2 sin alpha cos alpha)",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "introduced"
    },
    {
      "fact_id": "f_4",
      "text": "cot(2 alpha) = ((4/5)^2 - (3/5)^2) / (2 * (3/5) * (4/5))",
      "parent_fact_ids": ["f_1", "f_2", "f_3"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_5",
      "text": "cot(2 alpha) = 7/24",
      "parent_fact_ids": ["f_4"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Few-shot example 6: denominator nonzero condition only when actually needed

Problem:
Given x != 0 and x * y = 1. Find y.

Visible solution:
Since x is nonzero, divide both sides of x * y = 1 by x. Thus y = 1 / x.

Correct output:

Reasoning:
- Necessary given facts: x != 0 and x * y = 1.
- No introduced formula is needed.
- The nonzero condition is necessary to justify division by x.
- The final answer depends directly on the equation and the nonzero condition.

```json
{
  "problem_id": "example_6",
  "problem_text": "Given x != 0 and x * y = 1. Find y.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "x != 0",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "x * y = 1",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_3",
      "text": "y = 1 / x",
      "parent_fact_ids": ["f_1", "f_2"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}
```

Now convert the given math problem and its visible solution into an FDG.

Return:
1. A concise Reasoning section.
2. The final FDG JSON object inside the last ```json ... ``` code block.

Do not output any text after the final JSON code block.