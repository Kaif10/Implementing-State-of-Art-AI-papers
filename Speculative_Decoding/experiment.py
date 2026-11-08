"""
Reproduction experiment: do our measurements obey the paper's analysis?

Leviathan et al. (2023) give a closed form for the expected speedup of
speculative decoding (their Eq. for the "expected improvement factor"):

        speedup(K, alpha, c) =        1 - alpha^(K+1)
                               ------------------------------
                               (1 - alpha) * (c * K + 1)

  alpha = per-token acceptance rate (how often the target accepts a draft token)
  K     = draft length (the paper's gamma)
  c     = cost of one draft forward / one target forward (wall-clock)

This script, on a *modern* open LLM (Qwen2.5), measures alpha and the wall-clock
speedup across a sweep of K, independently measures c, and checks the measured
speedup against the formula. The estimate is idealized; agreement or disagreement must be interpreted with
its assumptions. Our cached implementation uses K+1 draft forwards per round,
so the displayed estimate accounts for that extra cache update.
"""

import json
import statistics
from pathlib import Path

import argparse
import sys
import time

import torch

# Windows consoles default to cp1252, which cannot print the Greek letters below.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import (autoregressive_generate_cached,
                                  speculative_generate_cached, _prime)
from benchmark import synchronize, timed, validate_tokenizers


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predicted_speedup(K, alpha, c, extra_draft_steps=0):
    """Idealized estimate; optionally account for our extra draft cache update."""
    if K < 1 or not 0 <= alpha <= 1 or c < 0:
        raise ValueError("expected K >= 1, alpha in [0, 1], c >= 0")
    # Geometric sum also handles perfect acceptance without 0/0.
    return sum(alpha ** i for i in range(K + 1)) / (c * (K + extra_draft_steps) + 1)


def measure_cost_ratio(target, draft, input_ids, reps=20):
    """Median single-token decode latency at the prompt length; prefill excluded."""
    def time_fwd(model):
        durations = []
        with torch.no_grad():
            for i in range(reps + 3):
                cache, anchor = _prime(model, input_ids)
                _, elapsed = timed(
                    lambda: model(anchor, past_key_values=cache, use_cache=True),
                    input_ids.device,
                )
                if i >= 3:
                    durations.append(elapsed)
        return statistics.median(durations)
    t_target, t_draft = time_fwd(target), time_fwd(draft)
    return t_draft / t_target, t_draft, t_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--prompt", default="The key idea behind speculative decoding is")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="experiment_results.json",
                    help="Save configuration and individual timings for audit")
    args = ap.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1 or any(k < 1 for k in args.Ks):
        ap.error("repeats, max-new-tokens and Ks must be positive")

    device = pick_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    print(f"device: {device}  dtype: {dtype}")
    print(f"target: {args.target}   draft: {args.draft}\n")

    tok = AutoTokenizer.from_pretrained(args.target)
    validate_tokenizers(tok, AutoTokenizer.from_pretrained(args.draft))
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    n = args.max_new_tokens

    # --- measure the cost ratio c ----------------------------------------- #
    c, td, tt = measure_cost_ratio(target, draft, ids)
    print(f"per-forward latency  draft={td*1e3:.1f} ms  target={tt*1e3:.1f} ms"
          f"  ->  c = {c:.3f}\n")

    # Warm up each path, then retain every synchronized timing.
    def measure(fn):
        fn()
        runs = []
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed + repeat)
            (_, stats), seconds = timed(fn, device)
            runs.append({"seconds": seconds, **stats})
        return runs

    baseline = measure(lambda: autoregressive_generate_cached(
        target, ids, n, temperature=args.temperature))
    base_t = statistics.median(r["seconds"] for r in baseline)
    results = []
    print(f"baseline median: {base_t:.2f}s ({n / base_t:.2f} tok/s)")
    print("K  acceptance  speedup  idealized(K+1 draft forwards)")
    for K in args.Ks:
        runs = measure(lambda: speculative_generate_cached(
            target, draft, ids, n, K=K, temperature=args.temperature))
        spec_t = statistics.median(r["seconds"] for r in runs)
        alpha = statistics.mean(r["acceptance_rate"] for r in runs)
        estimate = predicted_speedup(K, alpha, c, extra_draft_steps=1)
        row = {"K": K, "runs": runs, "median_seconds": spec_t,
               "acceptance_rate": alpha, "speedup": base_t / spec_t,
               "idealized_speedup": estimate}
        results.append(row)
        print(f"{K:2d} {alpha:10.3f} {base_t / spec_t:8.2f}x {estimate:12.2f}x")

    report = {"config": vars(args), "device": device, "dtype": str(dtype),
              "torch_version": torch.__version__, "cost_ratio": c,
              "draft_decode_seconds": td, "target_decode_seconds": tt,
              "baseline": baseline, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Raw timings saved to {args.output}")
    print("The estimate assumes constant acceptance and target verification cost.")
    print("Prefill, K-dependent verification, sampling and Python overhead can cause gaps.")
    print("These measurements alone do not identify which overhead dominates.")


if __name__ == "__main__":
    main()
