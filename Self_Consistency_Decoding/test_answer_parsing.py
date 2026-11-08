import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("self_consistency_parsing", Path(__file__).with_name("answer_parsing.py"))
parsing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parsing)


def test_equivalent_numbers_share_a_vote():
    assert {parsing.parse_arithmetic_first("The answer is " + x) for x in ["1000", "1,000.0", "1e3"]} == {"1000"}
    assert parsing.parse_arithmetic_first("The answer is 0.5.") == "1/2"
    assert parsing.parse_arithmetic_first("The answer is 1/2.") == "1/2"


def test_missing_and_invalid_answers_abstain():
    for text in ["no conclusion", "The answer is", "The answer is unknown", "The answer is 1/0"]:
        assert parsing.parse_arithmetic_first(text) is None
    assert parsing.parse_commonsense_first("The answer is") is None


def test_commonsense_normalizes_case_and_whitespace():
    assert parsing.parse_commonsense_first("The answer is  An Umbrella.") == "an umbrella"
