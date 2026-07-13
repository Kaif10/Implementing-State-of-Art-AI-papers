"""Evaluate a saved skill on a fresh held-out test set.

    python evaluate.py best_skill.md
    python evaluate.py best_skill.md --backend anthropic
"""

from __future__ import annotations

import argparse

from skillopt import (
    AnthropicBackend,
    NumberFormattingTask,
    SimulatedBackend,
    SkillDocument,
    SkillOpt,
    SkillOptConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SkillOpt skill.")
    parser.add_argument("skill_path")
    parser.add_argument("--backend", choices=["simulated", "anthropic"], default="simulated")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    backend = (
        SimulatedBackend()
        if args.backend == "simulated"
        else AnthropicBackend(model=args.model)
    )
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
