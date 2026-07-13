# s1: Budget Forcing for Test-Time Scaling

A tiny, honest reimplementation of the core inference-time idea from
**[s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)** (Muennighoff et al., 2025).

> Test-time scaling is the defining theme of 2025: instead of making models
> bigger, you let them *think longer at inference*. The s1 paper's most elegant
> contribution is **budget forcing** — a ~20-line decoding trick that lets you
> dial a reasoning model's chain-of-thought longer or shorter, with **zero
> retraining**.

This folder implements exactly that, and runs it on a small open reasoning model
so you can watch it work on your own machine.

## The idea in two moves

A reasoning model wraps its chain-of-thought in a delimiter — for
DeepSeek-R1-Distill that's `<think> ... </think>`. Budget forcing intervenes on
that delimiter while decoding:

| Move | How | Effect |
|------|-----|--------|
| **Lengthen** | When the model emits `</think>`, *suppress it* and append the word **"Wait"**. | The model second-guesses itself and keeps reasoning. More compute. |
| **Shorten** | If thinking exceeds a token budget, *force* `</think>` and make it answer now. | Caps compute. |

That's the whole paper's test-time trick. See `budget_forcing.py` — the logic
fits on one screen.

## Quickstart

```bash
cd S1_Test_Time_Scaling
uv venv && source .venv/bin/activate        # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # or: pip install -r requirements.txt

# Ask one question, forcing the model to think harder (inject "Wait" 4 times):
python run.py --num-ignore 4 --show-thinking "How many r's are in 'strawberry'?"

# Watch accuracy/behavior change as you scale test-time compute:
python demo.py
```

First run downloads `DeepSeek-R1-Distill-Qwen-1.5B` (~3.5 GB). It runs on CPU,
Apple Silicon (MPS), or CUDA — the code auto-detects. The 1.5B model is small
enough for a laptop; pass `--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (or
the original `simplescaling/s1.1-32B`) if you have the hardware.

## The knobs

- `--num-ignore N` — **lengthen**: how many times to suppress the end of
  thinking and inject "Wait". This is the primary test-time-scaling dial.
- `--max-thinking-tokens N` — **shorten**: hard cap on chain-of-thought length.
- `--sample` / `--temperature` — decoding (defaults to greedy for reproducibility).

## Files

```
budget_forcing.py   # the core: BudgetForcer.generate() with the two moves
run.py              # CLI for a single question
demo.py             # holds a question fixed, scales up thinking, shows the effect
```

## Honest scope

The s1 paper has **two** parts:

1. **s1K + SFT** — curating 1,000 high-quality reasoning traces and fine-tuning
   Qwen2.5-32B on them (a few GPU-hours).
2. **Budget forcing** — the inference-time trick above.

**This implementation covers (2) faithfully** and applies it to an off-the-shelf
open reasoning model, so it's runnable in a weekend on commodity hardware. It
does **not** retrain a model or reproduce the s1K dataset. Budget forcing is
model-agnostic; using a model that was already RL/SFT-trained to reason (like
R1-Distill) is what makes the "Wait" trick bite. If you want the full pipeline,
the authors' code is at <https://github.com/simplescaling/s1>.

## Citation

```bibtex
@article{muennighoff2025s1,
  title={s1: Simple test-time scaling},
  author={Muennighoff, Niklas and Yang, Zitong and Shi, Weijia and others},
  journal={arXiv preprint arXiv:2501.19393},
  year={2025}
}
```
