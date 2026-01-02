# Reflexion (minimal)

Paper: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

## Algorithmic flow (paper)

1) The actor attempts the task and produces an answer.
2) An evaluator judges the attempt and returns feedback.
3) A reflector summarizes lessons from the attempt and feedback.
4) The actor retries with a memory of recent reflections.
5) Repeat until the task passes or the trial limit is reached.
6) The actor receives the full reflection memory at each iteration (caution: memory growth increases input tokens).

## Exceptions in this repo

- CoT-based Reflexion only; the ReAct/tool-calling variant is not implemented yet.
- The evaluator is LLM-based judge instead of environment reward or task-specific tests as in the paper.
- Actor and evaluator outputs are parsed from JSON; failures fall back to raw text.
- Reflection memory is single-task only; the paper also explores cross-task memory which I personally thought was not really needed at a simpler level.

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
