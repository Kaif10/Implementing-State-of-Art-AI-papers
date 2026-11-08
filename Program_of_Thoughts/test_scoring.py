"""Offline regression tests for code extraction and numerical grading."""
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("pot_review_scoring", Path(__file__).with_name("program_of_thought.py"))
pot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pot
spec.loader.exec_module(pot)


@pytest.mark.parametrize("pred,gold", [
    ("1e3", "1000"), ("$1,000.00", "1000"), (".5", "1/2"),
    ("-3", "-3.0"), ("25%", "25"), ("0.1", "1/10"),
])
def test_numeric_equivalence(pred, gold):
    assert pot.is_correct(pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("1001", "1000"), ("9007199254740993", "9007199254740992"),
    ("1,00", "100"), ("answer 42 or 43", "42"), ("1/0", "1"),
    ("nan", "0"), ("inf", "0"), ("", "0"), (None, "0"),
])
def test_incorrect_or_ambiguous_answers_are_rejected(pred, gold):
    assert not pot.is_correct(pred, gold)


def test_absolute_tolerance_is_explicit():
    assert pot.is_correct("1.001", "1", tol=0.001)
    assert not pot.is_correct("1001", "1000", tol=0.001)


def test_code_extraction_and_assignment_fallback():
    assert pot.extract_code("```python\nanswer = 42\n```") == "answer = 42"
    assert pot.last_assigned_name("first = 1\nresult = first + 2") == "result"
    assert pot.last_assigned_name("invalid =") is None


def test_execution_reports_computed_value_and_failure():
    assert pot.execute_code("answer = 6 * 7") == ("42", "ok")
    assert pot.execute_code("total = 2 + 3") == ("5", "ok")
    assert pot.execute_code("raise ValueError('fixture')") == (None, "error")
    assert pot.execute_code("pass") == (None, "no_answer_var")


def test_execution_timeout():
    assert pot.execute_code("while True: pass", timeout=0.1) == (None, "timeout")
