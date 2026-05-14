You are an expert in mathematical reasoning and Fact Dependency Graph construction.

Your task is to convert a given math problem and its visible solution into a Fact Dependency Graph, abbreviated as FDG.

The FDG has only one node type: Fact.
Each Fact must be a minimal mathematical or logical assertion.

You must output exactly one valid JSON object.
Do not output markdown fences.
Do not output explanations.
Do not output a Reasoning section.
Do not output chain-of-thought.
Do not output <think>.
Do not output any text before or after the JSON object.

The JSON object must have exactly these top-level keys:
- "problem_id"
- "problem_text"
- "facts"

The value of "facts" must be a list of Fact objects.

Each Fact object must have exactly these keys:
- "fact_id"
- "text"
- "parent_fact_ids"
- "is_final_answer"
- "origin"

No other keys are allowed.

Use consecutive fact IDs:
"f_1", "f_2", "f_3", ...

Allowed origin values are exactly:
- "given"
- "introduced"
- "derived"
- "answer"

Origin rules:
1. Use origin="given" for necessary facts directly stated in the problem statement.
2. If a definition, formula, notation, custom operation, variable condition, or domain condition is explicitly stated in the problem statement, it must be origin="given".
3. Use origin="introduced" only for formulas, identities, helper definitions, or helper notation introduced by the solution and not stated in the problem.
4. Use origin="derived" for facts obtained by substitution, calculation, simplification, transformation, inference, or case selection.
5. Use origin="answer" only for the unique final mathematical answer.
6. Computed, transformed, substituted, simplified, or inferred facts must not be marked as origin="given".
7. There must be exactly one fact with is_final_answer=true.
8. The unique final answer fact must have origin="answer".
9. All non-answer facts must have is_final_answer=false.

Problem text rule:
- "problem_text" must copy the original problem statement, not the visible solution.
- Mathematical notation in facts may be normalized to plain ASCII.
- Prefer plain ASCII math in fact text, such as sqrt(x), x^2, <=, >=, !=, in.

Fact construction rules:
1. Each fact must be one minimal assertion.
2. Do not combine independent assertions into one fact.
3. Avoid long equality chains such as "x + 1 = 2 + 1 = 3".
4. Do not include raw solution narration or procedural language such as "we compute", "next", "then", "therefore", or "the answer is".
5. The final answer fact must be a mathematical assertion, such as "x = 2", "9 & 2 = 3sqrt(3)/4", or "cot(2 alpha) = 7/24".
6. If the requested value is an expression, include that expression in the final answer fact.
7. Do not add a wrapper fact like "The final answer is ...".
8. Omit facts that do not directly or indirectly contribute to the final answer.
9. Do not include facts only because they appear in the problem or solution; include only necessary facts.
10. Simple arithmetic simplification may be used directly, but the resulting fact must still be a single assertion.
11. Prefer candidate-set assertions such as "x in {-2, 2}" over multi-conclusion assertions such as "x = 2 or x = -2".

Dependency rules:
1. parent_fact_ids must only reference earlier facts.
2. The facts list must be topologically ordered.
3. The graph must be acyclic.
4. Every given fact must have parent_fact_ids=[].
5. Introduced facts may have parent_fact_ids=[] when they are standalone formulas, identities, definitions, or helper notation.
6. Every derived fact must have at least one parent.
7. Every answer fact must have at least one parent unless the answer is directly stated in the problem.
8. For each derived or answer fact, parent_fact_ids must be the nearest minimal sufficient direct premise set.
9. "Sufficient" means the listed parents are enough to derive the current fact.
10. "Minimal" means no listed parent is unnecessary.
11. "Direct" means use the closest earlier facts, not remote ancestors.
12. Do not include all problem conditions by default.
13. Do not repeat remote ancestors if their information is already contained in a nearer intermediate fact.
14. If a listed parent does not directly help derive the fact, remove it.
15. If a fact cannot be derived from its listed parents, add the missing nearest direct parent.

Implicit condition rules:
1. Include implicit conditions only when they are actually used to derive, select, justify, or disambiguate a later fact.
2. Do not introduce implicit conditions merely because they are generally true.
3. Useful implicit conditions may include positivity of a side length, denominator nonzero, square-root sign condition, or trigonometric sign condition.
4. An implicit condition is usually origin="derived" if it follows from earlier facts.
5. If a domain condition is explicitly stated in the problem, mark it as origin="given".

Special rule for custom definitions:
- If the problem defines a custom operation or notation, the definition must be included as a given fact whenever it is needed to derive the answer.
- Example: If the problem states "a * b = a + 2b", then the fact "a * b = a + 2b" has origin="given", not origin="introduced".

Correct example 1: irrelevant given facts omitted

Problem:
Given a = 1 and b = 2. Find a + 1.

Visible solution:
Since a = 1, a + 1 = 2.

Output:
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

Correct example 2: custom operation stated in the problem is given

Problem:
The operation * is defined by a * b = a + 2b. Find 3 * 4.

Visible solution:
Using the definition, 3 * 4 = 3 + 2 * 4 = 11.

Output:
{
  "problem_id": "example_custom_operation",
  "problem_text": "The operation * is defined by a * b = a + 2b. Find 3 * 4.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "a * b = a + 2b",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "given"
    },
    {
      "fact_id": "f_2",
      "text": "3 * 4 = 3 + 2 * 4",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_3",
      "text": "3 * 4 = 11",
      "parent_fact_ids": ["f_2"],
      "is_final_answer": true,
      "origin": "answer"
    }
  ]
}

Correct example 3: solver-introduced identity

Problem:
Given sin alpha = 3/5 and cos alpha = 4/5. Find cot(2 alpha).

Visible solution:
Using cot(2 alpha) = (cos^2 alpha - sin^2 alpha) / (2 sin alpha cos alpha), substitute sin alpha = 3/5 and cos alpha = 4/5. Then cot(2 alpha) = 7/24.

Output:
{
  "problem_id": "example_identity",
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

Correct example 4: implicit condition used for selection

Problem:
Given x > 0 and x^2 = 4. Find x.

Visible solution:
Since x^2 = 4, x = 2 or x = -2. Because x > 0, x = 2.

Output:
{
  "problem_id": "example_positive_selection",
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

Now convert the user-provided problem and visible solution into an FDG.
Return only the valid JSON object.