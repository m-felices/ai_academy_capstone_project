ANSWER_PROMPT = """
You are an AI assistant answering questions about a AI project.

If the answer can be inferred from ANY part of the context, answer it.
Only say "I don't know" if NO part of the context contains the answer.


Question:
{question}

Context:
{context}

Answer clearly.
"""
EVAL_PROMPT = """
Score the answer from 1–5 on:
- relevance
- faithfulness
- clarity

Context:
{context}

Answer:
{answer}

Return JSON only.
"""
REWRITE_PROMPT = """
You are an AI assistant improving clarity only.

Rules:
- Do NOT add new information
- Do NOT change facts
- Improve structure and wording only

Answer:
{answer}

Rewrite clearly.
"""
# REFLECTION_PROMPT = """
# Review the answer below.
#
# - Is it supported by the context?
# - Is anything missing or misleading?
# - Improve clarity.
#
# Context:
# {context}
#
# Answer:
# {answer}
#
# Return a revised answer.
# """