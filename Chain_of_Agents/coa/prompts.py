# Prompts taken from the paper (Tables 11–12).

WORKER_QUERY = """Worker {i}:
{chunk}
Here is the summary of the previous source text: {prev_cu}
Question: {query}
You need to read current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to answer the Query, if any. So please write the summary that can include the evidence for answering the Query:"""

MANAGER_QUERY = """{task_requirements}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_cu}
Question: {query}
Answer:"""

WORKER_NONQUERY = """Worker {i}:
{chunk}
Here is the summary of the previous source text: {prev_cu}
You need to read the current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to generate a summary for the whole text. Thus, your generated summary should be relatively long."""

MANAGER_NONQUERY = """{task_requirements}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_cu}
Answer:"""

# The paper does not publish an explicit judge prompt; this is a minimal default.
JUDGE_PROMPT = """You are given multiple candidate summaries (Communication Units) produced by different Chain-of-Agents paths.
Pick the candidate that is most reliable for answering the task, then answer using ONLY that candidate.

Task requirements:
{task_requirements}

Query (may be empty):
{query}

Candidates:
{candidates}

Return format:
- First line: selected id (e.g., "path_3")
- Remaining lines: final answer
"""
