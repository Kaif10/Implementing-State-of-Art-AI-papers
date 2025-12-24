# CAMEL (Role-Playing Agent) Implementation

This folder contains a compact implementation of the CAMEL paper. It follows the paper's role-playing setup with a task specifier, a user agent, and an assistant agent, plus the termination rules described in the paper.

## What matches the paper

- Task specifier prompt to refine the preliminary task
- Role-playing agents with fixed system prompts (AI Society and Code settings)
- Turn-by-turn conversation between User and Assistant agents
- Paper-style termination: max messages, user no-instruct rounds, `<CAMEL_TASK_DONE>`, role reversal, optional token limit

## Differences / caveats

- Uses OpenAI chat models instead of a finetuned or custom model
- Token counting is an estimate for budgeting
- Minimal CLI and logging (no evaluation harness)

## Quick start

1) Install deps (example)

```bash
python -m venv .venv
. .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install openai tiktoken
```

2) Set your API key

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3) Run AI Society mode

```bash
python run.py --mode ai_society --task "Draft a basic trading strategy" --assistant-role "Python Programmer" --user-role "Stock Trader"
```

4) Run Code mode

```bash
python run.py --mode code --task "Build a simple backtest function" --domain Finance --language Python
```

## How it works (high level)

- `prompts.py`: prompt templates from the paper (AI Society and Code)
- `agents.py`: chat model wrapper and role‑playing agent state
- `session.py`: task specification, parsing, and termination logic
- `run.py`: CLI entry point

## Notes

- Increase `--max-messages` or `--word-limit` for longer interactions.
