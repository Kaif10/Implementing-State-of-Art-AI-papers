# Reflexion (minimal)

Minimal end-to-end sketch of the Reflexion loop: an actor proposes, an evaluator judges, a reflector writes lessons, and the actor retries using a rolling memory.

## Files
- `core.py` - core roles and loop; actor outputs JSON with `trajectory` + `answer`, evaluator outputs JSON with `passed`/`must_fix`/`feedback`.

## Setup
1) Install deps (OpenAI Python SDK):
   ```bash
   pip install openai
   ```
2) Provide an API key (prefer env var):
   ```bash
   setx OPENAI_API_KEY "sk-..."
   ```

## Run the demo
```bash
python core.py
```
The script runs up to `trials` attempts (default 5) on the demo task in `core.py` and prints the final attempt.

## Implementation notes
- Actor sees task + context + recent reflections; returns a high-level trajectory (3-7 steps) and final answer.
- Evaluator judges only against task/context and returns JSON feedback.
- Reflector uses trajectory + evaluator feedback to write a concise lesson for the next attempt.
- Memory is a sliding window of the last `omega` reflections (default 3).

## Alignment with the Reflexion paper
- Implements the reasoning-style loop: attempt -> evaluate -> reflect -> retry.
- Evaluation is LLM-based here; swap the evaluator to use environment reward/tests to match specific benchmarks.
