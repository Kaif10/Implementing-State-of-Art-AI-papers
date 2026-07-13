"""The two models SkillOpt talks to.

Both implement the same tiny interface:

  answer(skill_text, question) -> str
      The FROZEN agent. Given the current skill and a question, it produces an
      answer. Its behavior changes only because the skill text changes.

  propose_edits(prompt, schema) -> dict
      The OPTIMIZER. Given a reflection prompt, it returns edits to the skill
      as JSON matching `schema`. (In the paper this is "a separate optimizer
      model".)

AnthropicBackend uses a real, frozen Claude model -- this is what produces
real results. SimulatedBackend is a deterministic, offline stand-in with no
network calls, so the full loop can be run and tested without an API key. The
simulator is NOT a language model: it is a transparent rule-based mock for the
built-in demo task only.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


def _first_text(response) -> str:
    """Pull the first text block out of an Anthropic response."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


class Backend(ABC):
    @abstractmethod
    def answer(self, skill_text: str, question: str) -> str:
        ...

    @abstractmethod
    def propose_edits(self, prompt: str, schema: dict) -> dict:
        ...


# --------------------------------------------------------------------------- #
# Real backend: a frozen Claude model.
# --------------------------------------------------------------------------- #
class AnthropicBackend(Backend):
    """Talks to a frozen Claude model via the Anthropic API.

    The same model serves as both the agent and the optimizer by default; pass
    `optimizer_model` to use a different model for the optimizer.
    """

    def __init__(self, model: str = "claude-opus-4-8", optimizer_model: str | None = None):
        import anthropic  # imported lazily so the simulator works without the SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.optimizer_model = optimizer_model or model

    def answer(self, skill_text: str, question: str) -> str:
        system = (
            "You answer questions. Follow the guidance in the SKILL section "
            "exactly. Reply with ONLY the final answer -- no explanation, no "
            "extra words.\n\n"
            "=== SKILL ===\n"
            f"{skill_text or '(no guidance yet)'}\n"
            "=== END SKILL ==="
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        return _first_text(response).strip()

    def propose_edits(self, prompt: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.optimizer_model,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(_first_text(response))


# --------------------------------------------------------------------------- #
# Offline backend: a deterministic mock for the demo task.
# --------------------------------------------------------------------------- #
_PERCENT = re.compile(r"(-?\d+\.?\d*)\s+percent of\s+(-?\d+\.?\d*)")
_PLUS = re.compile(r"(-?\d+\.?\d*)\s+plus\s+(-?\d+\.?\d*)")
_MINUS = re.compile(r"(-?\d+\.?\d*)\s+minus\s+(-?\d+\.?\d*)")
_TIMES = re.compile(r"(-?\d+\.?\d*)\s+times\s+(-?\d+\.?\d*)")


def _compute(question: str) -> float | None:
    q = question.lower()
    for pattern, fn in (
        (_PERCENT, lambda a, b: a / 100 * b),
        (_PLUS, lambda a, b: a + b),
        (_MINUS, lambda a, b: a - b),
        (_TIMES, lambda a, b: a * b),
    ):
        m = pattern.search(q)
        if m:
            return fn(float(m.group(1)), float(m.group(2)))
    return None


# Lines the simulated optimizer can add, keyed by the rule it detects as missing.
_RULE_LINES = {
    "decimals": "Round every answer to exactly two decimal places.",
    "usd": "Prefix the numeric answer with 'USD '.",
    "parentheses": "For negative results, write them in parentheses instead of a minus sign, e.g. (USD 5.00).",
}


class SimulatedBackend(Backend):
    """Deterministic, offline mock for the NumberFormattingTask demo.

    As an agent, it computes the value and then formats it -- but it only
    applies a house rule if the skill text mentions the matching cue. So adding
    the right guidance to the skill genuinely improves its answers, which is
    what lets the end-to-end loop converge without any network calls.

    As an optimizer, it reads the failures embedded in the reflection prompt,
    compares each wrong answer to the correct one, and proposes adding whichever
    rule line is missing.
    """

    # --- agent ---
    def answer(self, skill_text: str, question: str) -> str:
        value = _compute(question)
        if value is None:
            return "?"
        skill = (skill_text or "").lower()
        decimals = 2 if "two decimal" in skill else 1
        use_usd = "usd" in skill
        use_paren = "parenthes" in skill

        rounded = round(value, decimals)
        number = f"{abs(rounded):.{decimals}f}"
        prefix = "USD " if use_usd else ""

        if rounded < 0 and use_paren:
            return f"({prefix}{number})"
        if rounded < 0:
            return f"-{prefix}{number}"
        return f"{prefix}{number}"

    # --- optimizer ---
    def propose_edits(self, prompt: str, schema: dict) -> dict:
        # Only the CURRENT SKILL section counts as "already in the skill" --
        # the rest of the prompt echoes gold answers (which mention USD etc.).
        m = re.search(r"CURRENT SKILL:\s*(.*?)\n\nFAILURES", prompt, re.DOTALL)
        skill = (m.group(1) if m else "").lower()
        needed: list[str] = []

        pairs = re.findall(r'got:\s*"(.*?)"\s*correct:\s*"(.*?)"', prompt)
        for wrong, correct in pairs:
            # Two decimals: correct ends in .dd but the model's answer didn't.
            if re.search(r"\.\d\d\b", correct) and not re.search(r"\.\d\d\b", wrong):
                if "two decimal" not in skill:
                    needed.append("decimals")
            # Currency prefix.
            if "usd" in correct.lower() and "usd" not in wrong.lower():
                if "usd" not in skill:
                    needed.append("usd")
            # Negative-in-parentheses.
            if correct.startswith("(") and not wrong.startswith("("):
                if "parenthes" not in skill:
                    needed.append("parentheses")

        # De-duplicate while preserving order.
        seen: set[str] = set()
        operations = []
        for key in needed:
            if key not in seen:
                seen.add(key)
                operations.append({"action": "add", "index": 0, "text": _RULE_LINES[key]})
        return {"operations": operations}
