"""Train a skill end to end and save the deployable artifact.

Examples
--------
Local open-source model (no API key needed; downloads the model on first run):
    python train.py

Frozen OpenAI model via the API:
    export OPENAI_API_KEY=...
    python train.py --backend openai
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
    parser = argparse.ArgumentParser(description="Train a SkillOpt skill document.")
    parser.add_argument(
        "--backend",
        choices=["hf", "openai"],
        default="hf",
        help="'hf' runs an open-source model locally; 'openai' uses a frozen OpenAI model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name (default: Qwen/Qwen2.5-1.5B-Instruct for hf, gpt-4.1-mini for openai)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="hf only: model dtype (bfloat16 halves memory on low-RAM machines)",
    )
    parser.add_argument("--epochs", type=int, default=SkillOptConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=SkillOptConfig.batch_size)
    parser.add_argument("--train-size", type=int, default=SkillOptConfig.train_size)
    parser.add_argument("--val-size", type=int, default=SkillOptConfig.val_size)
    parser.add_argument("--test-size", type=int, default=SkillOptConfig.test_size)
    parser.add_argument("--learning-rate", type=int, default=SkillOptConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=SkillOptConfig.seed)
    parser.add_argument("--out", default="best_skill.md")
    args = parser.parse_args()

    config = SkillOptConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    if args.backend == "hf":
        kwargs = {"model": args.model} if args.model else {}
        backend = HuggingFaceBackend(dtype=args.dtype, **kwargs)
    else:
        backend = OpenAIBackend(**({"model": args.model} if args.model else {}))

    task = NumberFormattingTask()
    # Disjoint splits via different seeds.
    train = task.generate(config.train_size, seed=config.seed)
    val = task.generate(config.val_size, seed=config.seed + 1000)
    test = task.generate(config.test_size, seed=config.seed + 2000)

    print(f"task: {task.name} ({task.description})")
    print(f"backend: {args.backend}\n")

    optimizer = SkillOpt(backend, task, config)
    result = optimizer.train(train, val)

    # Honest before/after report on the held-out TEST set.
    no_skill = optimizer._score(SkillDocument(), test)
    with_skill = optimizer._score(result.best_skill, test)

    print("\n=== trained skill ===")
    print(result.best_skill.render())
    print("=== test results (held-out) ===")
    print(f"no skill:      {no_skill:.3f}")
    print(f"trained skill: {with_skill:.3f}")
    print(f"improvement:   {with_skill - no_skill:+.3f}")

    result.best_skill.save(args.out)
    print(f"\nsaved deployable skill to {args.out}")


if __name__ == "__main__":
    main()
