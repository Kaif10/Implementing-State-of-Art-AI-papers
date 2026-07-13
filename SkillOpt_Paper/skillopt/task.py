"""The built-in demo task.

A task is just: a way to make question/answer pairs, and a way to grade an
answer. The optimizer never sees the gold answers during a rollout -- it only
sees whether the model got each one right.

NumberFormattingTask is designed so that a *skill* is genuinely useful: the
math is easy, but the correct answer must follow three "house rules" that a
fresh model cannot guess. SkillOpt's job is to discover those rules from the
model's mistakes and write them into the skill document.

The three house rules (which the trained skill should end up encoding):
  1. Round to exactly two decimal places.
  2. Prefix the number with "USD ".
  3. Write negative results in parentheses instead of with a minus sign.

So 15 percent of 320  ->  "USD 48.00"
   12 minus 50        ->  "(USD 38.00)"
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Example:
    question: str
    answer: str


def format_gold(value: float) -> str:
    """Apply the three house rules to a numeric value."""
    rounded = round(value, 2)
    number = f"{abs(rounded):.2f}"
    if rounded < 0:
        return f"(USD {number})"
    return f"USD {number}"


class NumberFormattingTask:
    name = "number-formatting"
    description = (
        "Compute a simple arithmetic result and format it using a fixed set "
        "of house rules for currency answers."
    )

    def generate(self, n: int, seed: int) -> list[Example]:
        rng = random.Random(seed)
        examples: list[Example] = []
        for _ in range(n):
            kind = rng.choice(["percent", "plus", "minus", "times"])
            if kind == "percent":
                a = rng.randint(1, 95)
                b = rng.randint(20, 500)
                value = a / 100 * b
                q = f"What is {a} percent of {b}?"
            elif kind == "plus":
                a = rng.randint(1, 200)
                b = rng.randint(1, 200)
                value = a + b
                q = f"What is {a} plus {b}?"
            elif kind == "minus":
                # Bias toward negatives so the parenthesis rule matters.
                a = rng.randint(1, 100)
                b = rng.randint(1, 200)
                value = a - b
                q = f"What is {a} minus {b}?"
            else:  # times
                a = rng.randint(2, 25)
                b = rng.randint(2, 25)
                value = a * b
                q = f"What is {a} times {b}?"
            examples.append(Example(question=q, answer=format_gold(value)))
        return examples

    def grade(self, predicted: str, gold: str) -> bool:
        return predicted.strip() == gold.strip()
