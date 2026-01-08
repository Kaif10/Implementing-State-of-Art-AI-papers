# Multi-Agent Debate (MAD)

Paper: [Multi-Agent Debate](https://arxiv.org/search/?query=multi-agent+debate&searchtype=all&source=header)

This folder implements the Multi-Agent Debate setup: multiple independent agents answer the same task, then refine their answers over debate rounds by reading each others' responses. For comparable tasks, the final answer is selected by majority vote over parsed outputs.

## Algorithmic flow (paper)

1) Initialize N agents with an independent system prompt.
2) Round 1: each agent answers the task prompt (task-specific start prompt).
3) Rounds 2..R: for each agent, provide the other agents' last-round responses (concat or summarized) and ask to update its original answer based on the other answers it saw from the other agents.
4) If the task is comparable, parse final answers and early-stop when all agents agree.
5) Select the final output by majority vote over parsed answers; otherwise return the first agent's final response.

## Exceptions in this repo

- Uses OpenAI chat models and a minimal prompt set from the paper tables.
- Only the paper's core tasks are included: arithmetic, GSM8K, MMLU, biography, and chess.
- Chess outputs are lightly parsed; biography outputs are not merged by majority vote.
- Optional summarization of other-agent responses is a simple LLM summary prompt.

## Run the code

1) Install deps

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install openai python-dotenv
```

2) Set your API key

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3) Run a debate

```bash
python run_mad.py --task gsm8k --query "If John has 12 apples and gives away 5, how many remain?"
```

4) Run with summarization of other agents (optional)

```bash
python run_mad.py --task mmlu --query "What is the capital of Canada? (A) Toronto (B) Ottawa (C) Vancouver (D) Montreal" --other_mode summarize
```

5) Biography eval with LLM critic (optional)

```bash
python run_mad.py --task biography --query "Alan Turing" --bio_facts_file facts.txt --bio_person "Alan Turing"
```

## Files

- `mad_core.py`: prompts, parsing, debate loop, and majority vote
- `run_mad.py`: CLI entry point
- `mad_bio_eval.py`: optional biography evaluation (Appendix A.2-style)

