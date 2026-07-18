"""Evaluate a saved skill on a fresh held-out test set.

    python evaluate.py best_skill.md
    python evaluate.py best_skill.md --backend openai
"""

from __future__ import annotations

import argparse

from skillopt import (
    HuggingFaceBackend,
    NumberFormattingTask,
    OpenAIBackend,
    SkillDocument,
    SkillOpt,
    SkillOptConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SkillOpt skill.")
    parser.add_argument("skill_path")
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument(
        "--model",
        default=None,
        help="model name (default: Qwen/Qwen2.5-1.5B-Instruct for hf, gpt-4.1-mini for openai)",
    )
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.backend == "hf":
        backend = HuggingFaceBackend(**({"model": args.model} if args.model else {}))
    else:
        backend = OpenAIBackend(**({"model": args.model} if args.model else {}))
    task = NumberFormattingTask()
    test = task.generate(args.test_size, seed=args.seed)

    optimizer = SkillOpt(backend, task, SkillOptConfig())
    skill = SkillDocument.load(args.skill_path)

    no_skill = optimizer._score(SkillDocument(), test)
    with_skill = optimizer._score(skill, test)

    print(f"evaluated on {args.test_size} held-out questions")
    print(f"no skill:      {no_skill:.3f}")
    print(f"loaded skill:  {with_skill:.3f}")
    print(f"improvement:   {with_skill - no_skill:+.3f}")


if __name__ == "__main__":
    main()
