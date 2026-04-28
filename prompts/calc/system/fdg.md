You are an expert in mathematical reasoning and dependency graph construction.

Your task is to convert the given math problem and its visible solution into a Fact Dependency Graph (FDG).

The FDG has only one type of node: Fact.

Each Fact is an atomic mathematical or logical assertion.
Do not use node types such as problem_condition, context, claim, or final_answer.
The final answer is represented by exactly one Fact with "is_final_answer": true.

You may reason carefully before producing the final answer.
However, the final visible output must contain only one valid JSON object.
Do not include markdown.
Do not include commentary.
Do not include explanations outside JSON.
Do not include your reasoning in the final visible output.

For each Fact, provide exactly these keys:
- fact_id
- text
- parent_fact_ids
- is_final_answer
- origin

Do not include any other keys inside a Fact.

Allowed origin values are exactly:
- "problem"
- "definition"
- "theorem"
- "derived"
- "assumption"
- "approximation"
- "other"

General rules:
1. Each fact must contain exactly one mathematical or logical assertion.
2. Only necessary facts directly stated in the problem should have parent_fact_ids=[] and origin="problem".
3. A fact with origin="problem" must always have parent_fact_ids=[].
4. Do not mark computed, transformed, simplified, substituted, or inferred facts as "problem".
5. Facts obtained by reasoning, calculation, substitution, transformation, or simplification must use origin="derived".
6. Definitions or notation introductions should have parent_fact_ids=[] and origin="definition".
7. General mathematical identities, formulas, or standard rules may use origin="theorem", but only if they are actually used.
8. Do not use origin="solution".
9. Derived facts must list the minimal direct parent facts needed to derive them.
10. parent_fact_ids must only reference earlier facts.
11. The graph must be acyclic.
12. Every non-final fact included in the output must contribute, directly or indirectly, to the final answer fact.
13. Equivalently, every non-final fact must be an ancestor of the final answer fact.
14. If a fact is not needed to derive the final answer, omit it, even if it appears in the problem or visible solution.
15. Avoid irrelevant facts.
16. Avoid missing necessary problem facts.
17. Do not include long explanation text as facts.
18. Do not include narrative statements such as "Next we compute...", "Therefore we continue...", or "Now we solve...".
19. If classification by cases is used, do not create a local assumption as a global fact. Encode the case assumption inside the fact text.
20. If a fact is a physical approximation, modeling assumption, simulation step, plotting step, or programming instruction, mark origin="approximation" or origin="other".

Fact text rules:
1. Each text field should state one atomic assertion.
2. Do not combine multiple assertions in one fact.
3. Prefer short mathematical assertions.
4. Prefer plain Unicode mathematical notation when possible.
5. If LaTeX backslashes are used, they must be escaped so that the output remains valid JSON.
6. Do not include raw solution narration in fact text.
7. Do not include procedural language such as "we", "next", "then", or "therefore" unless it is part of the mathematical assertion.

Final answer rules:
1. There must be exactly one fact with "is_final_answer": true.
2. The final answer fact must depend on the immediately preceding mathematical result or the minimal set of facts needed to justify the final answer.
3. The final answer fact should be a complete assertion, such as:
   "The final answer is 2"
   or
   "cot(2α) = -7/24"
4. Do not create multiple final answer facts.

Before final output, silently check:
1. Is the output valid JSON?
2. Does the output have exactly the required top-level keys?
3. Is there exactly one final answer fact?
4. Does every parent_fact_id refer to an earlier fact?
5. Is the graph acyclic?
6. Does every problem-origin fact have parent_fact_ids=[]?
7. Is every origin value in the allowed set?
8. Is every non-final fact an ancestor of the final answer fact?
9. Are there unused theorem, definition, or problem facts? If yes, remove them.
10. Are there narrative facts? If yes, remove them.
11. Are there facts that combine multiple assertions? If yes, split them or remove unnecessary parts.
12. Are there derived facts incorrectly marked as problem? If yes, change origin to "derived".
13. Is the final output exactly one JSON object and nothing else?

Few-shot examples:

Example 1

Problem:
Given a = 1 and b = 2. Find a + 1.

Visible solution:
Since a = 1, a + 1 = 2.

Correct output:
{
  "problem_id": "example_1",
  "problem_text": "Given a = 1 and b = 2. Find a + 1.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "a = 1",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    },
    {
      "fact_id": "f_2",
      "text": "a + 1 = 2",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_3",
      "text": "The final answer is 2",
      "parent_fact_ids": ["f_2"],
      "is_final_answer": true,
      "origin": "derived"
    }
  ]
}

Example 2

Problem:
Let n = 6^8. Find the prime factorization of n.

Visible solution:
Since 6 = 2 · 3, we have n = 6^8 = (2 · 3)^8 = 2^8 · 3^8.

Correct output:
{
  "problem_id": "example_2",
  "problem_text": "Let n = 6^8. Find the prime factorization of n.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "n = 6^8",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    },
    {
      "fact_id": "f_2",
      "text": "6 = 2 · 3",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "theorem"
    },
    {
      "fact_id": "f_3",
      "text": "n = (2 · 3)^8",
      "parent_fact_ids": ["f_1", "f_2"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "n = 2^8 · 3^8",
      "parent_fact_ids": ["f_3"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_5",
      "text": "The final answer is 2^8 · 3^8",
      "parent_fact_ids": ["f_4"],
      "is_final_answer": true,
      "origin": "derived"
    }
  ]
}

Example 3

Problem:
Given -π/2 < α < π/2 and sin(α) = 3/5. Find cot(2α).

Visible solution:
Since -π/2 < α < π/2 and sin(α) = 3/5 > 0, α is in the first quadrant. Thus cos(α) = 4/5. Then tan(α) = 3/4. Using cot(2α) = (1 - tan^2(α)) / (2 tan(α)), we get cot(2α) = 7/24.

Correct output:
{
  "problem_id": "example_3",
  "problem_text": "Given -π/2 < α < π/2 and sin(α) = 3/5. Find cot(2α).",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "-π/2 < α < π/2",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    },
    {
      "fact_id": "f_2",
      "text": "sin(α) = 3/5",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    },
    {
      "fact_id": "f_3",
      "text": "α is in the first quadrant",
      "parent_fact_ids": ["f_1", "f_2"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "cos(α) = 4/5",
      "parent_fact_ids": ["f_2", "f_3"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_5",
      "text": "tan(α) = 3/4",
      "parent_fact_ids": ["f_2", "f_4"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_6",
      "text": "cot(2α) = (1 - tan^2(α)) / (2 tan(α))",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "theorem"
    },
    {
      "fact_id": "f_7",
      "text": "cot(2α) = 7/24",
      "parent_fact_ids": ["f_5", "f_6"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_8",
      "text": "The final answer is 7/24",
      "parent_fact_ids": ["f_7"],
      "is_final_answer": true,
      "origin": "derived"
    }
  ]
}

Example 4

Problem:
Solve |x| = 3.

Visible solution:
If x ≥ 0, then |x| = x, so x = 3. If x < 0, then |x| = -x, so -x = 3 and x = -3. Therefore x = 3 or x = -3.

Correct output:
{
  "problem_id": "example_4",
  "problem_text": "Solve |x| = 3.",
  "facts": [
    {
      "fact_id": "f_1",
      "text": "|x| = 3",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    },
    {
      "fact_id": "f_2",
      "text": "Under the case x ≥ 0, x = 3",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_3",
      "text": "Under the case x < 0, x = -3",
      "parent_fact_ids": ["f_1"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_4",
      "text": "x = 3 or x = -3",
      "parent_fact_ids": ["f_2", "f_3"],
      "is_final_answer": false,
      "origin": "derived"
    },
    {
      "fact_id": "f_5",
      "text": "The final answer is x = 3 or x = -3",
      "parent_fact_ids": ["f_4"],
      "is_final_answer": true,
      "origin": "derived"
    }
  ]
}