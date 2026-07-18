# SkillOpt — a minimal, honest reimplementation

A small Python implementation of the core idea in Microsoft's **SkillOpt**
paper, runnable fully locally with an open-source LLM:

> **SkillOpt: Executive Strategy for Self-Evolving Agent Skills**
> Yang et al., Microsoft Research (with SJTU, Tongji, Fudan), arXiv:2605.23904.
> Official code: https://github.com/microsoft/SkillOpt

This is **not** the official implementation. It's a clean-room, from-the-paper
reimplementation of the central mechanism, written to be readable end to end.
Where the paper has machinery this version doesn't, that's called out below
under [What this leaves out](#what-this-leaves-out).

---

## The one idea

> **The skill document is the parameter.**

You have a *frozen* language model — its weights never change. The thing you
train instead is a short piece of natural-language guidance (a "skill"). You
train it the same way you'd train weights:

| Weight training            | SkillOpt (text-space training)                          |
| -------------------------- | ------------------------------------------------------- |
| forward pass               | **rollout** — run the model on a batch, score answers   |
| gradient                   | **reflection** — an optimizer model reads the mistakes and writes edits |
| gradient step              | **edit** — apply a few add/delete/replace edits to the skill |
| learning rate              | **edit budget** — the max number of edits per step      |
| early stopping / val check | **validation gate** — keep an edit only if held-out score strictly improves |
| epochs, batches            | epochs, batches                                         |

The trained skill is a plain Markdown file (`best_skill.md`) you hand to the
model unchanged at deployment — **zero extra model calls at inference time.**

---

## The training loop (`skillopt/optimizer.py`)

Each step does four things, exactly as in the paper:

1. **Roll out** — run the frozen agent on a batch of training questions with
   the current skill; score each answer right/wrong.
2. **Reflect** — show the optimizer model the failures (and some successes) and
   ask for a few bounded edits. Recently-rejected edits are shown too, so it
   doesn't repeat them (the paper's *rejected-edit buffer*).
3. **Edit** — apply at most `learning_rate` operations to a *copy* of the skill.
4. **Gate** — score the copy on a held-out validation set. Keep it **only if it
   strictly beats** the current skill's validation score; otherwise discard it
   and remember it as a bad edit.

Repeat over several epochs. The best-validation skill is saved as the artifact.

---

## Quick start

### Local open-source model — no API key (the default)

```bash
pip install -r requirements.txt
python train.py                    # downloads Qwen2.5-1.5B-Instruct (~3 GB) on first run
python evaluate.py best_skill.md
```

This runs the whole loop end to end against a real, frozen open-source model
(`Qwen/Qwen2.5-1.5B-Instruct` by default — pass `--model` for any HF causal LM
with a chat template). It prints the training trace, the learned skill, a
held-out before/after comparison, and writes `best_skill.md`. Runs on CPU,
CUDA, or Apple Silicon (auto-detected); greedy decoding, so runs are
reproducible.

```bash
python tests/test_end_to_end.py   # or: python -m pytest -q
```

The logic tests need nothing installed; the end-to-end test uses the local
model and is skipped if torch/transformers are missing.

### Frozen OpenAI model via the API

```bash
export OPENAI_API_KEY=...
python train.py --backend openai --model gpt-4.1-mini
python evaluate.py best_skill.md --backend openai
```

Either way, the model's weights are never touched — only the skill document
changes.

---

## The built-in task

`NumberFormattingTask` asks easy arithmetic questions ("What is 15 percent of
320?") but the *correct* answer must follow three "house rules" a fresh model
can't guess:

1. round to exactly two decimal places,
2. prefix with `USD `,
3. write negatives in parentheses (`(USD 38.00)`).

A model with no skill gets these wrong; SkillOpt has to *discover the rules from
the model's mistakes* and write them into the skill. That makes the skill
causally responsible for the accuracy gain — the point of the paper. On the
offline demo it goes from 0% to 100% on a held-out test set.

Swap in your own task by providing `generate(n, seed) -> [Example]` and
`grade(predicted, gold) -> bool` (see `skillopt/task.py`).

---

## Files

```
skillopt/
  config.py      training knobs (epochs, batch size, learning rate, gate sizes)
  skill.py       the skill document: bounded add/delete/replace edits, render, save/load
  task.py        the task interface + the built-in NumberFormattingTask
  backends.py    the two models: HuggingFaceBackend (local open-source LLM) and OpenAIBackend (API)
  optimizer.py   the SkillOpt loop: rollout -> reflect -> edit -> gate
train.py         end-to-end: split data, train, report, save best_skill.md
evaluate.py      score a saved skill on a fresh test set (no-skill vs skill)
tests/           offline end-to-end tests
```

---

## How this maps to the paper, term by term

| Paper term                        | Here                                                            |
| --------------------------------- | --------------------------------------------------------------- |
| skill document = parameter        | `SkillDocument` — a list of guidance lines (`skill.py`)         |
| frozen agent                      | `Backend.answer(...)` — weights never change                    |
| separate optimizer model          | `Backend.propose_edits(...)`                                    |
| scored rollouts                   | `SkillOpt._rollout(...)`                                        |
| bounded add/delete/replace edits  | `SkillDocument.apply(operations, max_edits)`                    |
| textual learning-rate budget      | `config.learning_rate` (max edits per step)                     |
| held-out validation gate          | accept edit iff `candidate_val > working_val` (`optimizer.py`)  |
| rejected-edit buffer              | `rejected` list fed back into the reflection prompt             |
| epoch-wise slow / meta update     | `meta_notes` carried across steps; best-by-validation tracked   |
| deployable `best_skill.md`        | `SkillDocument.save(...)`; zero inference-time overhead         |

---

## What this leaves out

To stay minimal and readable, this implementation deliberately simplifies parts
of the paper. Being honest about the gaps:

- **Simplified "slow/meta update."** The paper describes an epoch-wise
  slow/meta update with an optimizer-side meta-skill memory. Here that's a
  lightweight `meta_notes` log plus the rejected-edit buffer; it is not a full
  separate meta-skill that's itself optimized.
- **One task, exact-match grading.** The paper spans six benchmarks, seven
  models, and three execution harnesses (direct chat, Codex CLI, Claude Code).
  This ships one self-contained demo task with exact-match scoring. The
  `Backend` / task interfaces are where you'd plug in real benchmarks and agent
  harnesses.
- **No agentic execution harness.** The agent here just answers a prompt. The
  paper also trains skills inside tool-using agent loops.
- **Small local models are noisy optimizers.** The reflection step asks the
  model to emit JSON edit operations; a 1.5B model occasionally produces
  malformed JSON. That is handled gracefully (an unparseable proposal is a
  no-op step, never a crash), but it means more steps get wasted than with a
  frontier API model. Use `--backend openai` or a larger `--model` for a
  stronger optimizer.
- **Reported numbers are this demo's, not the paper's.** The paper reports
  ~+19–25 point average gains across its grid. Gains here are specific to the
  toy task and the model you pick, and only show the *mechanism* working.

The goal is a faithful, legible skeleton of the algorithm — small enough to read
in one sitting — not a benchmark-grade reproduction.
