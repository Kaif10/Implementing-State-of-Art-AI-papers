from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol
from openai import OpenAI

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ChatModel(Protocol):
    def complete(self, messages: list[ChatMessage]) -> str: ...


@dataclass(frozen=True)
class OpenAIChatClient:
    """
    Minimal OpenAI chat wrapper.

    - One client instance (no per-call re-init)
    - Deterministic by default (temperature=0)
    """

    model: str
    api_key: str | None = None
    temperature: float = 0.0

    def complete(self, messages: list[ChatMessage]) -> str:
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


@dataclass
class ChatAgent:
    """
    Core CAMEL agent:
    """
    model: ChatModel
    system_prompt: str
    history: list[ChatMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history.append(ChatMessage(role="system", content=self.system_prompt))

    def add_counterpart(self, text: str) -> None:
        self.history.append(ChatMessage(role="user", content=text))

    def step(self) -> str:
        out = self.model.complete(self.history)
        self.history.append(ChatMessage(role="assistant", content=out))
        return out


class UserAgent(ChatAgent):
    pass


class AssistantAgent(ChatAgent):
    pass
