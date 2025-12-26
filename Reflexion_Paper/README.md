# Reflexion (minimal)

Paper: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

## Algorithmic flow (paper)

1) The actor attempts the task and produces an answer.
2) An evaluator judges the attempt and returns feedback.
3) A reflector summarizes lessons from the attempt and feedback.
4) The actor retries with a memory of recent reflections.
5) Repeat until the task passes or the trial limit is reached.

## Exceptions in this repo

- CoT-based Reflexion only; the ReAct variant is not implemented yet.
- The evaluator is LLM-based instead of environment reward or task-specific tests.
- Actor and evaluator outputs are parsed from JSON; failures fall back to raw text.

## Run the code

1) Install deps

```bash
pip install openai
```

2) Set your API key

Edit `core.py` and replace `your_key_here`, or remove that line and set:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3) Run the demo

```bash
python core.py
```

## Files

- `core.py`: core roles and Reflexion loop
