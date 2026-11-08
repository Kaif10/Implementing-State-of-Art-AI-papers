"""Checks for the SkillOpt loop.

The logic tests run offline with pytest. The end-to-end training
test runs the REAL loop with a local open-source model (small config, greedy
decoding) -- it needs torch/transformers installed and downloads the model on
first run. It is skipped unless RUN_MODEL_TESTS=1 and dependencies are installed.

Run with:  python -m pytest -q   (or)   python tests/test_end_to_end.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt import (
    NumberFormattingTask,
    SkillDocument,
    SkillOpt,
    SkillOptConfig,
)
from skillopt.task import Example, format_gold, generate_disjoint_splits

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

    Checks validation selection mechanics, not a guaranteed test accuracy gain.
    Set RUN_MODEL_TESTS=1 to opt into model loading and possible downloads.
    """
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("set RUN_MODEL_TESTS=1 to run the local-model integration test")
    if not _HAS_HF:
        pytest.skip("torch/transformers not installed")

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

    train, val, test = generate_disjoint_splits(task, (6, 4, 6), seed=0)

    result = optimizer.train(train, val, verbose=False)

    no_skill = optimizer._score(SkillDocument(), test)
    with_skill = optimizer._score(result.best_skill, test)
    assert 0 <= no_skill <= 1 and 0 <= with_skill <= 1

    # Every accepted edit must have strictly improved validation (the gate).
    val_after_accepts = [r.candidate_val_score for r in result.history if r.accepted]
    assert all(a < b for a, b in zip(val_after_accepts, val_after_accepts[1:]))


def test_validation_gate_accepts_improvement_and_rejects_ties_and_regressions():
    class ScriptedBackend:
        def __init__(self):
            self.proposals = iter(["good", "equivalent", "bad"])
            self.prompts = []

        def answer(self, skill_text, question):
            return "yes" if question == "validation" and any(
                rule in skill_text for rule in ("good", "equivalent")
            ) else "wrong"

        def propose_edits(self, prompt, schema):
            self.prompts.append(prompt)
            rule = next(self.proposals)
            action = "add" if rule == "good" else "replace"
            return {"operations": [{"action": action, "index": 0, "text": rule}]}

    backend = ScriptedBackend()
    optimizer = SkillOpt(backend, NumberFormattingTask(), SkillOptConfig(epochs=3, batch_size=1))
    result = optimizer.train([Example("train", "yes")], [Example("validation", "yes")], verbose=False)
    assert [r.accepted for r in result.history] == [True, False, False]
    assert result.best_skill.lines == ["good"]
    assert result.best_val_score == 1.0
    assert "equivalent" in backend.prompts[-1]  # rejected edit reaches the next reflection


def test_split_questions_are_unique_and_reproducible():
    splits = generate_disjoint_splits(NumberFormattingTask(), (40, 20, 40), seed=0)
    assert [len(s) for s in splits] == [40, 20, 40]
    questions = [ex.question for split in splits for ex in split]
    assert len(questions) == len(set(questions))
    assert splits == generate_disjoint_splits(NumberFormattingTask(), (40, 20, 40), seed=0)


def test_malformed_edit_fields_are_ignored():
    skill = SkillDocument(["one"])
    assert skill.apply([None, 42, {"action": 5}, {"action": "add", "text": ["bad"]}], 4) == skill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
