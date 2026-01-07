# Chain-of-Agents (CoA)

Paper: [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818)

A pretty interesting paper that helps solve a big problem that a single LLM might face while dealing huge context-> needle in a haystack, to a great extent

## Algorithmic flow (paper)

1) Split the long input into sentences and greedily chunk them under a token budget `k` (Algorithm 2).
2) Run a chain of worker agents over chunks to produce Communication Units (CUs) in sequence (Algorithm 1).
3) Pass the final CU to a manager agent to synthesize the final answer.
4) Optionally run multi-path variants (bidir, perm5, self5) to explore different orderings or sampling.
5) Select the final output by majority vote or a judge model.

## Exceptions in this repo
- No paper datasets or evaluation harness are included.

## Run the code

1) Install deps

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

2) Set your API key

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3) Run a single-path chain

```bash
python run.py --input_file doc_test.txt --task_requirements_file req_qa.txt --query "Summarize the key constraints"
```

4) Run a multi-path variant (optional)

```bash
python run.py --input_file doc_test.txt --task_requirements_file req_qa.txt --query "Summarize the key constraints" --multipath bidir --selector vote
```

## Files

- `coa/chunking.py`: Algorithm 2
- `coa/coa.py`: Algorithm 1 and multipath orchestration
- `coa/prompts.py`: worker/manager prompts from paper tables
- `coa/llm.py`: OpenAI Responses API wrapper and token counting
- `run.py`: CLI entry point
