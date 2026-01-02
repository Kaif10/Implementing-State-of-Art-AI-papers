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
JUDGE_PROMPT = """You are the judge in Multi-path Chain-of-Agents.

You are given multiple final Communication Units (CU_l), each produced by a different CoA path.
Your job:
1) Select the single CU_l that is MOST RELIABLE for completing the task.
2) Generate the final answer using ONLY the selected CU_l (do not use any other candidates).

Task requirements:
{task_requirements}

Query (may be empty):
{query}

Candidates (each is a final CU_l):
{candidates}

Return format:
First line: selected id (e.g., "path_3")
Remaining lines: final answer
"""

