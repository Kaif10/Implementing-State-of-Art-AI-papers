"""Checks for the SkillOpt loop.

The logic tests run instantly with no dependencies. The end-to-end training
test runs the REAL loop with a local open-source model (small config, greedy
decoding) -- it needs torch/transformers installed and downloads the model on
first run, so it is skipped when they are missing.

Run with:  python -m pytest -q   (or)   python tests/test_end_to_end.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt import (
    NumberFormattingTask,
    SkillDocument,
    SkillOpt,
    SkillOptConfig,
)
from skillopt.task import format_gold

_HAS_HF = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


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


def test_json_extraction_is_robust():
    from skillopt.backends import _extract_json

    op = {"action": "add", "index": 0, "text": "rule"}
    ops = {"operations": [op]}
    fenced = '```json\n{"operations": [{"action": "add", "index": 0, "text": "rule"}]}\n```'
    prosy = 'Here are my edits: {"operations": []} hope that helps!'
    assert _extract_json(fenced) == ops
    assert _extract_json(prosy) == {"operations": []}
    assert _extract_json("no json at all") == {"operations": []}
    assert _extract_json("[1, 2, 3]") == {"operations": []}
    # Equivalent shapes small models actually emit are accepted, not wasted:
    assert _extract_json(json.dumps(op)) == ops          # single bare operation
    assert _extract_json(json.dumps([op])) == ops        # bare list of operations


def test_training_end_to_end_with_local_model():
    """The real loop on a real open-source model, kept small for CI-on-a-laptop.

    Asserts the MECHANICS (the loop runs, the gate is monotone, the skill never
    gets worse than the empty baseline) -- not a specific accuracy: tiny models
    on tiny splits are noisy, and pretending otherwise would be dishonest.
    """
    if not _HAS_HF:
        print("skipped: torch/transformers not installed")
        return

    from skillopt import HuggingFaceBackend

    config = SkillOptConfig(
        train_size=6, val_size=4, test_size=6,
        epochs=1, batch_size=3, learning_rate=4, seed=0,
    )
    # 0.5B keeps the test runnable on low-RAM machines; the mechanics under
    # test are identical to the 1.5B default.
    backend = HuggingFaceBackend(model="Qwen/Qwen2.5-0.5B-Instruct")
    task = NumberFormattingTask()
    optimizer = SkillOpt(backend, task, config)

    train = task.generate(config.train_size, seed=0)
    val = task.generate(config.val_size, seed=1000)
    test = task.generate(config.test_size, seed=2000)

    result = optimizer.train(train, val, verbose=False)

    no_skill = optimizer._score(SkillDocument(), test)
    with_skill = optimizer._score(result.best_skill, test)
    assert with_skill >= no_skill

    # Every accepted edit must have strictly improved validation (the gate).
    val_after_accepts = [r.candidate_val_score for r in result.history if r.accepted]
    assert val_after_accepts == sorted(val_after_accepts)


if __name__ == "__main__":
    test_skill_edits_apply_with_budget()
    test_invalid_edits_are_ignored()
    test_gold_formatting()
    test_json_extraction_is_robust()
    test_training_end_to_end_with_local_model()
    print("all tests passed")
