"""Live cached-decoding correctness demonstration on a small local checkpoint.

By default the same checkpoint serves as draft and target to demonstrate exact
greedy agreement and the all-accepted path. This is not a speedup benchmark.
Use benchmark.py with distinct compatible checkpoints to measure performance.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding import autoregressive_generate_cached, speculative_generate_cached
from benchmark import validate_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--draft-model", default=None)
    args = parser.parse_args()
    print("Cached speculative decoding / live model correctness demo", flush=True)
    print(f"Target: {args.model}\nDraft: {args.draft_model or args.model}", flush=True)
    print("Same-checkpoint default demonstrates acceptance and exact output agreement.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    if args.draft_model and args.draft_model != args.model:
        validate_tokenizers(tokenizer, AutoTokenizer.from_pretrained(args.draft_model))
        draft = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=torch.float32).eval()
    else:
        draft = target
    prompt = "Write a Python function that adds two numbers."
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"Prompt: {prompt}\nGenerating 24 tokens with both paths...", flush=True)
    reference, _ = autoregressive_generate_cached(target, ids, 24, temperature=0)
    actual, stats = speculative_generate_cached(target, draft, ids, 24, K=3, temperature=0)
    same = torch.equal(reference, actual)
    print("\nGENERATED CONTINUATION:\n" + tokenizer.decode(actual[0, ids.shape[1]:], skip_special_tokens=True), flush=True)
    print(f"\nExact greedy agreement: {same}\nAccepted / examined proposals: {stats['accepted']} / {stats['proposed']}", flush=True)
    print(f"Target verification rounds: {stats['target_forwards']}\nAutoregressive decode forwards: 24", flush=True)
    if not same:
        raise SystemExit("Greedy outputs differed; inspect the recorded run")
    print("DEMO COMPLETE / correctness example, not a speedup measurement", flush=True)


if __name__ == "__main__":
    main()
