# Implementing SOTA AI Papers

Readable Python implementations of research ideas in LLM inference, reasoning,
retrieval, and agent coordination. Each project makes its core algorithm explicit
and includes paper-specific notes and runnable examples.

## Start here

- **[Speculative decoding](Speculative_Decoding)**: draft/verify sampling, KV-cache
  management, correctness checks, and a latency experiment.
- **[Program of Thoughts](Program_of_Thoughts)**: generate Python solutions, execute
  them, and evaluate answers on GSM8K with per-example output records.
- **[SkillOpt](SkillOpt_Paper)**: optimize a skill document with a frozen model,
  bounded edits, and validation-based selection.

The collection includes both algorithm demonstrations and evaluation scripts.
Project READMEs describe the implemented scope, model choices, and commands.

## Implemented papers

| Folder | Paper |
|--------|-------|
| [CAMEL_Paper](CAMEL_Paper) | [CAMEL: Communicative Agents for "Mind" Exploration of LLM Society](https://arxiv.org/abs/2303.17760) |
| [Chain_of_Agents](Chain_of_Agents) | [Chain of Agents: LLMs Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818) |
| [FLARE-Active_RAG](FLARE-Active_RAG) | [Active Retrieval Augmented Generation (FLARE)](https://arxiv.org/abs/2305.06983) |
| [IRCOT-multihop](IRCOT-multihop) | [Interleaving Retrieval with Chain-of-Thought for Multi-Step QA (IRCoT)](https://arxiv.org/abs/2212.10509) |
| [Multiagent_Debate](Multiagent_Debate) | [Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/abs/2305.14325) |
| [Program_of_Thoughts](Program_of_Thoughts) | [Program of Thoughts Prompting](https://arxiv.org/abs/2211.12588) |
| [Reflexion_Paper](Reflexion_Paper) | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) |
| [S1_Test_Time_Scaling](S1_Test_Time_Scaling) | [s1: Simple Test-Time Scaling (budget forcing)](https://arxiv.org/abs/2501.19393) |
| [Self-RAG](Self-RAG) | [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) |
| [Self_Consistency_Decoding](Self_Consistency_Decoding) | [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) |
| [SkillOpt_Paper](SkillOpt_Paper) | [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904) |
| [Speculative_Decoding](Speculative_Decoding) | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) |

## Setup and tests

Use **Python 3.12**. Install a project's `requirements.txt` in a virtual environment
(or follow its README for notebook-specific setup). API-backed examples read
`OPENAI_API_KEY` from the environment; local model examples document their model
and hardware requirements.

Run the lightweight regression suite from the repository root:

```bash
python -m pip install -r requirements-test.txt
python scripts/run_tests.py
```

Include the decoding tests with CPU PyTorch and Transformers installed:

```bash
python -m pip install -r requirements-test-models.txt
python scripts/run_tests.py --include-models
```

These tests exercise scoring, data separation, prompt budgets, agent state,
retrieval decisions, and decoding mechanics using fixtures and small synthetic
models. They require no API keys or pretrained model downloads. GitHub Actions
runs the complete offline suite. Live-model experiments are separate commands
in the project READMEs.

## Reproducing an experiment

Record the command, commit, dataset split, model revision, sampling settings,
software versions, hardware, and seed alongside the output. Preserve per-example
predictions and raw timings so comparisons can be inspected. Use the same examples
and account for all model calls when comparing methods.

## Project layout

Each folder is self-contained, with its own paper link, implementation notes,
and entry point. The implementations prioritize readable core mechanisms;
API-based adaptations and inference-only implementations are identified in their
project documentation.
