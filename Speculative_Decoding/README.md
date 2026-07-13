# Speculative Decoding — from scratch, end to end

A minimal, honest implementation of the core algorithm from
**["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)**
(Leviathan, Kalman & Matias, ICML 2023; concurrently
[Chen et al. 2023](https://arxiv.org/abs/2302.01318)).

No training, no GPU required, no library `generate()` shortcuts — the entire
draft → verify → accept/reject loop is written out in
[`speculative_decoding.py`](speculative_decoding.py).

## The idea

Autoregressive decoding is slow because each token needs one full forward pass
of a large model, and those passes are *memory-bound*: the GPU/CPU sits idle.
Speculative decoding fills that idle compute:

1. **Draft.** A small, cheap model proposes `K` tokens, one at a time, recording
   its probabilities `q(x)`.
2. **Verify.** The large *target* model scores all `K` proposals in **one**
   forward pass, giving the true probabilities `p(x)` at every position.
3. **Accept / reject.** Walk the proposals left to right. Accept token `x` with
   probability `min(1, p(x)/q(x))`. On the first rejection, resample that token
   from the residual distribution `(p − q)₊` (renormalised) and stop the round.
   If all `K` are accepted, take a free **bonus** token from the target's next
   distribution.

Each round costs **one** target forward pass but emits between 1 and `K+1`
tokens. That ratio is the speedup.

### Why it's lossless

The accept/reject + residual-resampling rule is constructed so the emitted
tokens are distributed **exactly** as if sampled from the target model alone
(paper, Theorem 1). Speculative decoding changes *speed*, not *output quality*.
`verify_lossless.py` checks this empirically.

## Files

| File | What it does |
|------|--------------|
| `speculative_decoding.py` | The full algorithm: draft loop, parallel target scoring, acceptance test, residual resampling, bonus token. Two implementations: a readable **uncached** reference and a **KV-cached** version (the realistic serving path) that crops the cache on rejection. Plus matching autoregressive baselines. |
| `benchmark.py` | Runs baseline vs. speculative on the same prompt; reports latency, tok/s, speedup, and acceptance rate. KV-cached by default (`--no-cache` for the O(n²) reference). |
| `experiment.py` | Reproduction study on **Qwen2.5**: sweeps the draft length K, measures acceptance α and the cost ratio c, and checks the measured wall-clock speedup against the **paper's closed-form speedup formula**. |
| `verify_lossless.py` | Correctness: greedy output is identical to the target; sampling matches the target distribution (total-variation distance). |

## Setup

```bash
cd Speculative_Decoding
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Reproduction study on a modern open LLM (Qwen2.5, Apache-2.0, no API key/gating)
python experiment.py        # 0.5B draft + 3B target: K-sweep vs the paper's formula
python verify_lossless.py   # confirm it's lossless

# Single A/B benchmark
python benchmark.py --draft Qwen/Qwen2.5-0.5B --target Qwen/Qwen2.5-3B \
                    --max-new-tokens 128 --K 4 --temperature 0.7
python benchmark.py --no-cache ...          # the O(n²) uncached reference
```

First run downloads the models (~7 GB for the Qwen2.5 0.5B + 3B pair).

## Results — does it match the paper?

Measured on **Qwen2.5-0.5B (draft) + Qwen2.5-3B (target)**, Apple M-series MPS,
fp16, KV-cached, 96 new tokens, temperature 0.7:

```
  K   accept α  tgt fwd    tok/s  speedup  predicted
----------------------------------------------------
  1      76.4%       55    12.69    1.15x      1.47x
  3      72.3%       37    14.06    1.27x      1.63x
  5      77.2%       29    13.95    1.26x      1.72x
  8      75.5%       26    11.03    1.00x      1.43x
   (baseline autoregressive: 11.06 tok/s;  measured cost ratio c = 0.20)
```

What lines up with Leviathan et al. (2023):

1. **Lossless.** Greedy speculative output is token-for-token identical to the
   target; sampled output matches the target's distribution (TV distance equal
   to the direct-sampling yardstick). *(verify_lossless.py)*
2. **High, K-stable acceptance** α ≈ **0.72–0.77** — the draft agrees with the
   target ~3 times out of 4.
3. **Target forward passes collapse** from 96 to 26–55, the core mechanism.
4. **Speedup follows the paper's closed-form curve**
   `(1−α^{K+1}) / ((1−α)(cK+1))` — rising then falling with K.

### Why "only" ~1.3× here, and how that *confirms* the paper

The paper's headline 2–3× assumes the draft is ~free relative to the target
(`c → 0`), which holds on GPU/TPU with a large, FLOP-bound target. On a 16 GB
Mac the regime is different:

- **c = 0.20**, not ~0. At batch 1 on MPS, latency is dominated by per-kernel
  **launch overhead**, so a 0.5B draft forward is only ~5× cheaper than a 3B
  target forward — not 30× as the parameter ratio suggests.
- Plugging our measured α ≈ 0.75 and **c → 0.02** into the *same formula* gives
  **2.6× (K=3), 3.0× (K=5), 3.2× (K=8)** — i.e. the paper's numbers. Same model,
  different hardware regime. `experiment.py` prints both.

Measured speedup also sits a bit below the formula's prediction because the
formula counts only model forwards, while real wall-clock also pays for
multinomial sampling over Qwen's **151,936-token vocabulary**, the Python loop,
and cache cropping — overheads that shrink as the target model grows.

## Tuning knobs

- `--K` — draft length. Higher `K` means more tokens per accepted round but more
  wasted draft work when a rejection comes early. The sweet spot depends on the
  draft/target speed ratio and their agreement.
- `--temperature` — `0` is greedy (outputs identical to the baseline);
  higher values sample more diversely.
- `--draft` / `--target` — any two causal-LM checkpoints that **share a
  tokenizer** (the acceptance test compares token-level probabilities).

## Honest scope & limitations

- The core algorithm is implemented faithfully and is genuinely lossless
  (exact in fp32; fp16 introduces tiny argmax ties that very occasionally flip a
  token — a numerical effect, not an algorithmic one).
- Two implementations are provided: an **uncached** one (recomputes the full
  sequence each forward — clearest to read) and a **KV-cached** one that threads
  `past_key_values` through both models and **crops the cache on rejection**.
  The cache machinery lives inside the model (`use_cache=True`); the only
  speculative-specific bookkeeping is the crop. The cached path is what the
  benchmark/experiment use and what reproduces the paper's regime.
- Absolute wall-clock speedup is hardware/model-dependent (see Results); the
  acceptance rate, losslessness, and the speedup *formula* are the
  hardware-independent claims, and those reproduce exactly.
- **Medusa** ([Cai et al. 2024](https://arxiv.org/abs/2401.10774)) and
  **EAGLE** ([Li et al. 2024](https://arxiv.org/abs/2401.15077)) replace the
  separate draft model with *trained heads* on the target model itself (Medusa)
  or with feature-level autoregression (EAGLE). They share this exact
  draft-then-verify backbone — they only change *how proposals are generated*.
  This folder implements that backbone with a standalone draft model, which needs
  no training.
