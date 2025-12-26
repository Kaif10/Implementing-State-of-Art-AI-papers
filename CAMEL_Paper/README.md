# CAMEL (Role-Playing Agents)

Paper: [CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760)

## Algorithmic flow (paper)

1) Use a task specifier to refine a preliminary task into a clear instruction with roles and constraints.
2) Initialize the User and Assistant role prompts for the selected mode (AI Society or Code).
3) Alternate turns between User and Assistant, conditioning each response on the role prompt and conversation history.
4) Stop when a termination rule fires: max messages, user no-instruct rounds, `<CAMEL_TASK_DONE>`, role reversal, or token limit.


## Run the code

1) Install deps

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

## Files

- `prompts.py`: prompt templates from the paper (AI Society and Code)
- `agents.py`: chat model wrapper and role-playing agent state
- `session.py`: task specification, parsing, and termination logic
- `run.py`: CLI entry point
