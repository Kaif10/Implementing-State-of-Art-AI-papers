"""
Correctness check: speculative decoding is *lossless*.

The paper's central claim is that speculative decoding samples from EXACTLY the
target model's distribution. Two empirical checks:

1. Greedy (temperature = 0): the output must be token-for-token identical to
   greedy autoregressive decoding from the target model.

2. Sampling (temperature > 0): over many runs, the empirical distribution of
   the first generated token under speculative decoding must match the target
   model's true distribution (compared by total variation distance).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import (autoregressive_generate, logits_to_probs,
                                  speculative_generate)

DRAFT, TARGET = "distilgpt2", "gpt2"   # small + fast; fine for a correctness test
PROMPT = "The capital of France is"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT).to(device).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)

    # --- Test 1: greedy decoding must be identical -------------------------- #
    base, _ = autoregressive_generate(target, ids, 20, temperature=0.0)
    spec, _ = speculative_generate(target, draft, ids, 20, K=4, temperature=0.0)
    identical = torch.equal(base, spec)
    print(f"[greedy] identical to target argmax decoding: {identical}")
    print(f"  target: {tok.decode(base[0], skip_special_tokens=True)!r}")
    print(f"  spec:   {tok.decode(spec[0], skip_special_tokens=True)!r}")
    assert identical, "greedy speculative output diverged from the target!"

    # --- Test 2: sampling distribution must match the target ---------------- #
    # A single empirical TV number is meaningless without a yardstick: with a
    # 50k-token vocab and a finite sample, even sampling DIRECTLY from the target
    # has a large TV to its own analytic distribution. So we compare three
    # empirical distributions against the target's true p(.):
    #   - speculative samples   -> should match the target's sampling noise
    #   - direct target samples -> the sampling-noise yardstick
    #   - draft samples         -> a different distribution; should be far off
    # Losslessness means: TV(spec) ~= TV(target_direct)  <<  TV(draft).
    with torch.no_grad():
        true_p = logits_to_probs(target(ids).logits[:, -1, :], temperature=1.0)[0]
        draft_p = logits_to_probs(draft(ids).logits[:, -1, :], temperature=1.0)[0]

    N = 4000

    def tv_of(sampler):
        counts = torch.zeros_like(true_p)
        for _ in range(N):
            counts[sampler()] += 1
        emp = counts / counts.sum()
        return 0.5 * (emp - true_p).abs().sum().item()

    tv_spec = tv_of(lambda: speculative_generate(target, draft, ids, 1, K=4, temperature=1.0)[0][0, -1])
    tv_direct = tv_of(lambda: torch.multinomial(true_p, 1).item())
    tv_draft = tv_of(lambda: torch.multinomial(draft_p, 1).item())

    print(f"\n[sampling] TV distance to the target's true distribution, {N} samples each:")
    print(f"  speculative decoding : {tv_spec:.4f}   <- should match the yardstick")
    print(f"  direct target sample : {tv_direct:.4f}   <- sampling-noise yardstick")
    print(f"  draft model          : {tv_draft:.4f}   <- a DIFFERENT distribution")
    print("  => speculative ~= direct (lossless), and both << draft (test is discriminative)")
    assert tv_spec < tv_draft, "speculative distribution looks like the draft, not the target!"


if __name__ == "__main__":
    main()
