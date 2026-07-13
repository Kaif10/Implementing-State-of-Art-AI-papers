"""
run.py — ask a single question with a chosen thinking budget.

Examples:
    # default budget
    python run.py "If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?"

    # LENGTHEN: force the model to think harder (inject "Wait" 4 times)
    python run.py --num-ignore 4 "How many r's are in 'strawberry'?"

    # SHORTEN: cap the chain of thought at 200 tokens
    python run.py --max-thinking-tokens 200 "Prove sqrt(2) is irrational."
"""

import argparse

from budget_forcing import BudgetForcer


def main():
    p = argparse.ArgumentParser(description="s1 budget forcing on an open reasoning model")
    p.add_argument("question", help="the question to ask")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--num-ignore", type=int, default=0,
                   help="LENGTHEN: number of times to suppress </think> and inject 'Wait'")
    p.add_argument("--max-thinking-tokens", type=int, default=4096,
                   help="SHORTEN: hard cap on thinking tokens")
    p.add_argument("--answer-max-tokens", type=int, default=512)
    p.add_argument("--wait-phrase", default="Wait")
    p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--show-thinking", action="store_true", help="print the chain of thought")
    args = p.parse_args()

    forcer = BudgetForcer(model_name=args.model)
    res = forcer.generate(
        args.question,
        max_thinking_tokens=args.max_thinking_tokens,
        num_ignore=args.num_ignore,
        wait_phrase=args.wait_phrase,
        answer_max_tokens=args.answer_max_tokens,
        sample=args.sample,
        temperature=args.temperature,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print(f"Q: {res.question}")
    print("-" * 70)
    if args.show_thinking:
        print("THINKING:\n" + res.thinking + "\n" + "-" * 70)
    print("ANSWER:\n" + res.answer)
    print("-" * 70)
    print(f"thinking_tokens={res.thinking_tokens}  "
          f"waits_injected={res.waits_injected}  capped={res.capped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
