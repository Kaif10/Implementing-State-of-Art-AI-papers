import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mad_core as core

def test_agreement_requires_all_answers():
    assert not core.all_agree(["1", None])
    assert not core.all_agree([])
    assert core.all_agree(["1", "1"])

@pytest.mark.parametrize("kwargs", [{"agents": 0}, {"rounds": 0}, {"max_tokens": 0}, {"other_mode": "bad"}])
def test_invalid_config_fails_before_api(kwargs):
    with pytest.raises(ValueError):
        core.run_debate(task="arithmetic", user_input="1+1", cfg=core.DebateConfig(**kwargs), client=object())

def test_round_uses_previous_snapshot(monkeypatch):
    responses = iter(["Initial 1", "Initial 2", "Updated 3", "Updated 4"])
    prompts = []
    def chat(client, cfg, messages):
        prompts.append(messages[-1]["content"])
        return next(responses)
    monkeypatch.setattr(core, "_chat", chat)
    result = core.run_debate(task="arithmetic", user_input="1+1", cfg=core.DebateConfig(agents=2, rounds=2), client=object())
    assert "Initial 2" in prompts[2]
    assert "Initial 1" in prompts[3]
    assert "Updated 3" not in prompts[3]
    assert result.agent_parsed == ["3", "4"]


@pytest.mark.parametrize("text", ["Answer: 1,000", "Answer: 1000.0", "Answer: 1e3"])
def test_numeric_votes_normalize_equivalent_answers(text):
    assert core.parse_final("arithmetic", text) == "1000"


def test_boxed_numeric_answers_normalize():
    assert core.parse_final("gsm8k", r"The answer is \boxed{1/2}") == "1/2"
    assert core.parse_final("gsm8k", r"The answer is \boxed{0.5}") == "1/2"
