You are an expert in mathematical reasoning and dependency graph construction.

Your task is to convert the given math problem and its visible solution into a Fact Dependency Graph (FDG).

The FDG has only one type of node: Fact.

Each Fact is an atomic mathematical or logical assertion.
The final answer is represented by exactly one Fact with "is_final_answer": true and origin="answer".

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
- "given": a necessary fact directly stated in the problem
- "introduced": notation, variables, definitions, or intermediate objects introduced for the solution
- "derived": a fact obtained by reasoning, calculation, substitution, transformation, or simplification
- "answer": the single final answer fact

General rules:
1. Each fact must contain exactly one mathematical or logical assertion.
2. Only necessary facts directly stated in the problem should have parent_fact_ids=[] and origin="given".
3. A fact with origin="given" must always have parent_fact_ids=[].
4. Do not mark computed, transformed, simplified, substituted, or inferred facts as "given".
5. Facts obtained by reasoning, calculation, substitution, transformation, or simplification must use origin="derived".
6. A fact with origin="derived" must list the minimal direct parent facts needed to derive it.
7. Use origin="introduced" for notation introductions, helper variables, definitions, named intermediate quantities, or setup facts that are introduced rather than directly stated as final mathematical consequences.
8. Use origin="answer" only for the single final answer fact, and set is_final_answer=true for that fact.
9. Do not use origin values such as "problem", "definition", "theorem", "assumption", "approximation", "other", or "solution".
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
20. If a physical approximation, modeling assumption, simulation step, plotting step, or programming instruction is necessary, encode it as an introduced or derived fact using the minimal parents.

Fact text rules:
1. Each text field should state one atomic assertion.
2. Do not combine multiple assertions in one fact.
3. Prefer short mathematical assertions.
4. Prefer plain mathematical notation when possible.
5. If LaTeX backslashes are used, they must be escaped so that the output remains valid JSON.
6. Do not include raw solution narration in fact text.
7. Do not include procedural language such as "we", "next", "then", or "therefore" unless it is part of the mathematical assertion.

Final answer rules:
1. There must be exactly one fact with "is_final_answer": true.
2. The final answer fact must have origin="answer".
3. No other fact may have origin="answer".
4. The final answer fact must depend on the immediately preceding mathematical result or the minimal set of facts needed to justify the final answer.
5. The final answer fact should be a complete assertion, such as:
   "The final answer is 2"
   or
   "cot(2 alpha) = -7/24"

Before final output, silently check:
1. Is the output valid JSON?
2. Does the output have exactly the required top-level keys?
3. Is there exactly one final answer fact?
4. Does the final answer fact have origin="answer"?
5. Does every parent_fact_id refer to an earlier fact?
6. Is the graph acyclic?
7. Does every given fact have parent_fact_ids=[]?
8. Does every derived fact have at least one parent?
9. Is every origin value in the allowed set?
10. Is every non-final fact an ancestor of the final answer fact?
11. Are there unused given, introduced, or derived facts? If yes, remove them.
12. Are there narrative facts? If yes, remove them.
13. Are there facts that combine multiple assertions? If yes, split them or remove unnecessary parts.
14. Is the final output exactly one JSON object and nothing else?

Few-shot example:

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
      "origin": "given"
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
      "origin": "answer"
    }
  ]
}
