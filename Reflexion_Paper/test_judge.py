"""Judge responses must fail closed when their types violate the schema."""
import importlib.util
from pathlib import Path
import sys
import json

import pytest

spec = importlib.util.spec_from_file_location("reflexion_test_core", Path(__file__).with_name("core.py"))
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def call(self, *args, **kwargs):
        return self.response


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_only_json_boolean_is_accepted(value):
    result = core.judge(FakeLLM(json.dumps({"passed": value})), core.Task("task"), "answer")
    assert result.passed is False


def test_valid_boolean_and_feedback():
    result = core.judge(FakeLLM('{"passed":true,"feedback":"ok","must_fix":[]}'), core.Task("task"), "answer")
    assert result.passed is True
    result = core.judge(FakeLLM('{"passed":false,"feedback":"missing","must_fix":["include total"]}'), core.Task("task"), "answer")
    assert not result.passed and "include total" in result.feedback


@pytest.mark.parametrize("response", ['not json', '{"passed":true,"feedback":[]}', '{"passed":true,"must_fix":"fix"}'])
def test_malformed_judgments_fail_closed(response):
    assert not core.judge(FakeLLM(response), core.Task("task"), "answer").passed
