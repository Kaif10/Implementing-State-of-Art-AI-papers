"""
demo.py — the money shot: hold the question fixed and scale up test-time compute.

We ask the same question while injecting more and more "Wait"s (num_ignore).
The paper's central claim is that letting the model think longer at inference
time tends to improve its answers — without changing a single weight.

    python demo.py
    python demo.py --question "How many r's are in 'strawberry'?"
"""

import argparse

from budget_forcing import BudgetForcer

DEFAULT_Q = (
    "I have 6 cups of flour. A loaf needs 2 cups and a cake needs 3 cups. "
    "I want to bake at least one of each and use as much flour as possible. "
    "How many loaves and cakes should I bake, and how much flour is left over?"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", default=DEFAULT_Q)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 2, 4],
                   help="num_ignore values (how many 'Wait's) to compare")
    p.add_argument("--max-thinking-tokens", type=int, default=4096)
    args = p.parse_args()

    forcer = BudgetForcer(model_name=args.model)

    print(f"\nQUESTION:\n{args.question}\n")
    print("Scaling test-time compute by forcing the model to think longer:\n")

    for n in args.budgets:
        res = forcer.generate(
            args.question,
            num_ignore=n,
            max_thinking_tokens=args.max_thinking_tokens,
        )
        print("=" * 70)
        print(f"num_ignore={n}  ->  thinking_tokens={res.thinking_tokens}  "
              f"waits_injected={res.waits_injected}")
        print(f"ANSWER: {res.answer}")
    print("=" * 70)
    print("\nMore forced thinking == more test-time compute. That's budget forcing.")


if __name__ == "__main__":
    main()
