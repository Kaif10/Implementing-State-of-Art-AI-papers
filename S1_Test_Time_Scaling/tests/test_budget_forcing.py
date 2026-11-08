"""Scripted budget-forcing tests; no model or tokenizer downloads."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget_forcing import BudgetForcer


def scripted(segments):
    obj = object.__new__(BudgetForcer)
    obj.think_open, obj.think_close = "<think>", "</think>"
    obj._build_prefix = lambda *args: "<think>"
    calls = []
    def generate(text, budget, *args, **kwargs):
        calls.append((text, budget, kwargs))
        return segments.pop(0)
    obj._generate_segment = generate
    return obj, calls


def test_extends_then_answers_without_thinking_stop():
    obj, calls = scripted([("first</think>", 3), ("second</think>", 4), ("42", 1)])
    result = obj.generate("question", max_thinking_tokens=10, num_ignore=1)
    assert result.waits_injected == 1
    assert result.thinking_tokens == 7
    assert "Wait" in result.thinking and result.answer == "42"
    assert calls[1][1] == 7
    assert calls[-1][2] == {"stop_thinking": False}


def test_exhaustion_forces_answer():
    obj, calls = scripted([("unfinished", 3), ("answer", 1)])
    result = obj.generate("question", max_thinking_tokens=3)
    assert result.capped and result.thinking_tokens == 3
    assert "</think>" in calls[-1][0]


@pytest.mark.parametrize("kwargs", [{"max_thinking_tokens": 0}, {"answer_max_tokens": 0},
                                    {"num_ignore": -1}, {"num_ignore": 1, "wait_phrase": ""}])
def test_invalid_budgets(kwargs):
    obj, _ = scripted([])
    with pytest.raises(ValueError):
        obj.generate("question", **kwargs)


def test_delimiter_is_preserved_during_decoding():
    obj = object.__new__(BudgetForcer)
    obj.device, obj.think_close = "cpu", "</think>"
    class Tokenizer:
        eos_token, eos_token_id = "<eos>", 9
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1]]))
        def decode(self, ids, skip_special_tokens):
            return "reasoning" if skip_special_tokens else "reasoning</think>"
    obj.tokenizer = Tokenizer()
    obj.model = SimpleNamespace(generate=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]))
    segment, count = obj._generate_segment("prompt", 4, False, 0.6, 0.95)
    assert segment == "reasoning</think>" and count == 2
