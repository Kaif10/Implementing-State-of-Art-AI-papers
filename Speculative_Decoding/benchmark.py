"""
Benchmark: ordinary autoregressive decoding vs. speculative decoding.

Loads a small draft model and a large target model that SHARE A TOKENIZER
(required — the acceptance test compares token-level probabilities), generates
the same continuation with both methods under identical sampling settings, and
reports wall-clock latency, tokens/sec, the achieved speedup, and the draft
acceptance rate.

Default pair (runs on a Mac CPU/MPS, no GPU or training needed):
    draft  = gpt2        (124M)
    target = gpt2-large  (774M)

Example:
    python benchmark.py --max-new-tokens 128 --K 4 --temperature 0.7
    python benchmark.py --draft distilgpt2 --target gpt2-large
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import (autoregressive_generate,
                                  autoregressive_generate_cached,
                                  speculative_generate,
                                  speculative_generate_cached)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description="Speculative decoding benchmark")
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--target", default="gpt2-large")
    ap.add_argument("--prompt", default="The future of artificial intelligence is")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--K", type=int, default=4, help="draft length per round")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None,
                    help="float16 / bfloat16 / float32 (default: float16 on mps/cuda, float32 on cpu)")
    ap.add_argument("--no-cache", action="store_true",
                    help="use the simple O(n^2) uncached path instead of the KV-cached one")
    args = ap.parse_args()

    ar_fn = autoregressive_generate if args.no_cache else autoregressive_generate_cached
    spec_fn = speculative_generate if args.no_cache else speculative_generate_cached

    device = args.device or pick_device()
    dtype_name = args.dtype or ("float16" if device in ("mps", "cuda") else "float32")
    dtype = getattr(torch, dtype_name)
    print(f"device: {device}  dtype: {dtype_name}")

    tok = AutoTokenizer.from_pretrained(args.target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    print(f"loading draft:  {args.draft}")
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()

    # Sanity check: shared vocabulary is a hard requirement for the algorithm.
    assert target.config.vocab_size == draft.config.vocab_size, (
        "draft and target must share a tokenizer / vocabulary")

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    gen_kwargs = dict(temperature=args.temperature, top_k=args.top_k)

    def run_baseline():
        torch.manual_seed(args.seed)
        return ar_fn(target, input_ids, args.max_new_tokens, **gen_kwargs)

    def run_speculative():
        torch.manual_seed(args.seed)
        return spec_fn(target, draft, input_ids, args.max_new_tokens,
                       K=args.K, **gen_kwargs)

    print(f"\nmode: {'uncached (O(n^2))' if args.no_cache else 'KV-cached'}")
    print("warming up...")
    ar_fn(target, input_ids, 2, **gen_kwargs)
    spec_fn(target, draft, input_ids, 2, K=args.K, **gen_kwargs)

    print("running baseline (autoregressive, target only)...")
    (base_ids, _), base_t = timed(run_baseline)

    print("running speculative decoding...")
    (spec_ids, stats), spec_t = timed(run_speculative)

    n = args.max_new_tokens
    print("\n" + "=" * 64)
    print(f"prompt: {args.prompt!r}")
    print("=" * 64)
    print("\n--- baseline output ---")
    print(tok.decode(base_ids[0], skip_special_tokens=True))
    print("\n--- speculative output ---")
    print(tok.decode(spec_ids[0], skip_special_tokens=True))

    print("\n" + "=" * 64)
    print(f"{'method':<26}{'time (s)':>12}{'tok/s':>12}")
    print("-" * 64)
    print(f"{'autoregressive':<26}{base_t:>12.2f}{n / base_t:>12.2f}")
    print(f"{'speculative':<26}{spec_t:>12.2f}{n / spec_t:>12.2f}")
    print("-" * 64)
    print(f"speedup:                {base_t / spec_t:>6.2f}x")
    print(f"draft acceptance rate:  {stats['acceptance_rate'] * 100:>6.1f}%  "
          f"({stats['accepted']}/{stats['proposed']} drafted tokens accepted)")
    print(f"target forward passes:  {stats['target_forwards']}  "
          f"(vs {n} for autoregressive)")
    print(f"K (draft length):       {stats['K']}")
    print("=" * 64)
    print("\nNote: speculative decoding is LOSSLESS — both methods sample from the")
    print("same target distribution. Outputs differ only due to RNG ordering, not")
    print("quality. With temperature 0 (greedy) the two outputs are identical.")


if __name__ == "__main__":
    main()
