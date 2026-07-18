"""The two models SkillOpt talks to.

Both implement the same tiny interface:

  answer(skill_text, question) -> str
      The FROZEN agent. Given the current skill and a question, it produces an
      answer. Its behavior changes only because the skill text changes.

  propose_edits(prompt, schema) -> dict
      The OPTIMIZER. Given a reflection prompt, it returns edits to the skill
      as JSON matching `schema`. (In the paper this is "a separate optimizer
      model".)

HuggingFaceBackend runs an open-source instruct model locally (default
Qwen2.5-1.5B-Instruct) -- fully end to end on your own machine, no API key.
OpenAIBackend uses a frozen OpenAI model via the API. In both cases the
model's weights are never touched; only the skill document changes.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

_ANSWER_SYSTEM = (
    "You answer questions. Follow the guidance in the SKILL section "
    "exactly. Reply with ONLY the final answer -- no explanation, no "
    "extra words.\n\n"
    "=== SKILL ===\n"
    "{skill}\n"
    "=== END SKILL ==="
)

_JSON_SYSTEM = (
    "You improve skill documents. Reply with ONLY a JSON object matching the "
    "requested shape -- no prose, no markdown fences."
)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction; returns a no-op proposal on failure.

    Local instruct models occasionally wrap JSON in fences or prose. A failed
    parse must not crash training -- an unusable proposal is simply a no-op
    step, and the loop moves on.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"operations": []}


class Backend(ABC):
    @abstractmethod
    def answer(self, skill_text: str, question: str) -> str:
        ...

    @abstractmethod
    def propose_edits(self, prompt: str, schema: dict) -> dict:
        ...


# --------------------------------------------------------------------------- #
# Local backend: an open-source instruct model via Hugging Face transformers.
# --------------------------------------------------------------------------- #
class HuggingFaceBackend(Backend):
    """Runs a local open-source instruct model as both agent and optimizer.

    Any causal LM with a chat template works; the default is small enough for
    a laptop. Decoding is greedy, so runs are reproducible.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()

    def answer(self, skill_text: str, question: str) -> str:
        system = _ANSWER_SYSTEM.format(skill=skill_text or "(no guidance yet)")
        return self._generate(system, question, max_new_tokens=64)

    def propose_edits(self, prompt: str, schema: dict) -> dict:
        user = f"{prompt}\n\nThe JSON must match this schema:\n{json.dumps(schema)}"
        return _extract_json(self._generate(_JSON_SYSTEM, user, max_new_tokens=512))


# --------------------------------------------------------------------------- #
# API backend: a frozen OpenAI model.
# --------------------------------------------------------------------------- #
class OpenAIBackend(Backend):
    """Talks to a frozen OpenAI model via the API (needs OPENAI_API_KEY).

    The same model serves as both the agent and the optimizer by default; pass
    `optimizer_model` to use a different model for the optimizer.
    """

    def __init__(self, model: str = "gpt-4.1-mini", optimizer_model: str | None = None):
        import openai  # imported lazily so the local backend works without the SDK

        self.client = openai.OpenAI()
        self.model = model
        self.optimizer_model = optimizer_model or model

    def answer(self, skill_text: str, question: str) -> str:
        system = _ANSWER_SYSTEM.format(skill=skill_text or "(no guidance yet)")
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def propose_edits(self, prompt: str, schema: dict) -> dict:
        response = self.client.chat.completions.create(
            model=self.optimizer_model,
            max_completion_tokens=4000,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "skill_edits", "schema": schema, "strict": True},
            },
            messages=[
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return _extract_json(response.choices[0].message.content or "")
