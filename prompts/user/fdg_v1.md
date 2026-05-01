Problem:
{problem_text}

Solution:
{solution_or_cot}

Return the FDG JSON with this exact top-level structure:
{{
  "problem_id": "...",
  "problem_text": "...",
  "facts": [
    {{
      "fact_id": "f_1",
      "text": "...",
      "parent_fact_ids": [],
      "is_final_answer": false,
      "origin": "problem"
    }}
  ]
}}
