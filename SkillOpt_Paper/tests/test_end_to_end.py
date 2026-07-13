"""End-to-end checks that run fully offline (no API key).

Run with:  python -m pytest -q   (or)   python tests/test_end_to_end.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt import (
    NumberFormattingTask,
    SimulatedBackend,
    SkillDocument,
    SkillOpt,
    SkillOptConfig,
)
from skillopt.task import format_gold


def test_skill_edits_apply_with_budget():
    skill = SkillDocument(["one", "two", "three"])
    ops = [
        {"action": "replace", "index": 0, "text": "ONE"},
        {"action": "delete", "index": 2, "text": ""},
        {"action": "add", "index": 0, "text": "four"},
        {"action": "add", "index": 0, "text": "ignored-over-budget"},
    ]
    # Learning rate of 3 means the 4th edit is dropped.
    out = skill.apply(ops, max_edits=3)
    assert out.lines == ["ONE", "two", "four"]


def test_invalid_edits_are_ignored():
    skill = SkillDocument(["a"])
    ops = [
        {"action": "delete", "index": 99, "text": ""},   # out of range
        {"action": "replace", "index": 0, "text": ""},   # empty replacement
        {"action": "nonsense", "index": 0, "text": "x"}, # unknown action
    ]
    assert skill.apply(ops, max_edits=10).lines == ["a"]


def test_gold_formatting():
    assert format_gold(48.0) == "USD 48.00"
    assert format_gold(-38.0) == "(USD 38.00)"
    assert format_gold(7.5) == "USD 7.50"


def test_training_improves_accuracy():
    config = SkillOptConfig(
        train_size=40, val_size=20, test_size=60,
        epochs=3, batch_size=8, learning_rate=4, seed=0,
    )
    backend = SimulatedBackend()
    task = NumberFormattingTask()
    optimizer = SkillOpt(backend, task, config)

    train = task.generate(config.train_size, seed=0)
    val = task.generate(config.val_size, seed=1000)
    test = task.generate(config.test_size, seed=2000)

    result = optimizer.train(train, val, verbose=False)

    no_skill = optimizer._score(SkillDocument(), test)
    with_skill = optimizer._score(result.best_skill, test)

    # The skill should learn the house rules and reach perfect accuracy,
    # while the empty baseline does not.
    assert with_skill > no_skill
    assert with_skill == 1.0

    # Every accepted edit must have strictly improved validation (the gate).
    val_after_accepts = [r.candidate_val_score for r in result.history if r.accepted]
    assert val_after_accepts == sorted(val_after_accepts)


if __name__ == "__main__":
    test_skill_edits_apply_with_budget()
    test_invalid_edits_are_ignored()
    test_gold_formatting()
    test_training_improves_accuracy()
    print("all tests passed")
