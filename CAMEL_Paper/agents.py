from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ChatModel(Protocol):
    def complete(self, messages: list[ChatMessage]) -> str: ...


@dataclass(frozen=True)
class OpenAIChatClient:
    model: str
    api_key: str

    def complete(self, messages: list[ChatMessage]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""


def count_tokens_tiktoken(model: str, messages: list[ChatMessage]) -> int:
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    model_l = (model or "").lower()

    tokens_per_message = None
    tokens_per_name = None

    if (
        model_l.startswith("gpt-3.5-turbo")
        or model_l.startswith("gpt-4")
        or model_l.startswith("gpt-4o")
        or model_l.startswith("gpt-4.1")
        or model_l.startswith("o1")
        or model_l.startswith("o3")
    ):
        tokens_per_message = 3
        tokens_per_name = 1

    if tokens_per_message is None:
        return sum(len(enc.encode(m.content)) for m in messages)

    total = 0
    for m in messages:
        total += tokens_per_message
        total += len(enc.encode(m.role))
        total += len(enc.encode(m.content or ""))
        total += tokens_per_name

    total += 3
    return total


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
