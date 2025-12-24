from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from agents import AssistantAgent, ChatAgent, ChatModel, UserAgent, count_tokens_tiktoken
from prompts import (
    CAMEL_TASK_DONE,
    fill_angle_brackets,
    AI_SOCIETY_TASK_SPECIFIER_PROMPT,
    AI_SOCIETY_ASSISTANT_SYSTEM_PROMPT,
    AI_SOCIETY_USER_SYSTEM_PROMPT,
    CODE_TASK_SPECIFIER_PROMPT,
    CODE_ASSISTANT_SYSTEM_PROMPT,
    CODE_USER_SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass(frozen=True)
class Instruction:
    instruction: str
    input: str


def is_task_done(text: str) -> bool:
    return text.strip() == CAMEL_TASK_DONE


def parse_strict_instruction(text: str) -> Instruction | None:
    """
    Paper format:
      Instruction: ...
      Input: ...
    Exactly two non-empty lines.
    """
    if is_task_done(text):
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != 2:
        return None
    if not lines[0].startswith("Instruction:"):
        return None
    if not lines[1].startswith("Input:"):
        return None

    inst = lines[0][len("Instruction:") :].strip()
    inp = lines[1][len("Input:") :].strip()
    if not inst or not inp:
        return None

    return Instruction(instruction=inst, input=inp)


_INSTRUCTION_RE = re.compile(r"^\s*Instruction\s*:\s*(.+?)\s*$", re.IGNORECASE)
_INPUT_RE = re.compile(r"^\s*Input\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_instruction_relaxed(text: str) -> Instruction | None:
    """

    The paper's "User No Instruct" is conceptual ("user does not instruct assistant"),
    but LLM outputs often include extra whitespace/labels/commentary. Treat the user as
    "instructing" iff BOTH an Instruction: line and an Input: line appear anywhere.

    Extraction: returns the first occurrences (case-insensitive). Keeps "None" literal.
    """
    if is_task_done(text):
        return None

    inst = None
    inp = None
    for ln in text.splitlines():
        if inst is None:
            m = _INSTRUCTION_RE.match(ln)
            if m:
                inst = m.group(1).strip()
                continue
        if inp is None:
            m = _INPUT_RE.match(ln)
            if m:
                inp = m.group(1).strip()
                continue
        if inst is not None and inp is not None:
            break

    if not inst or not inp:
        return None
    return Instruction(instruction=inst, input=inp)


def assistant_is_instructing(text: str) -> bool:
    """
    Termination: assistant begins issuing 'Instruction:' (role reversal).
    Check first non-empty line.
    """
    for ln in (x.strip() for x in text.splitlines()):
        if ln:
            return ln.startswith("Instruction:")
    return False


@dataclass(frozen=True)
class TerminationConfig:
    max_messages: int = 40
    user_no_instruct_rounds: int = 3
    token_limit: int | None = None
    token_model: str | None = None


@dataclass(frozen=True)
class AiSocietyConfig:
    assistant_role: str
    user_role: str
    preliminary_task: str
    word_limit: int = 50


@dataclass(frozen=True)
class CodeConfig:
    domain: str
    language: str
    preliminary_task: str
    word_limit: int = 50


def _specify_task(model: ChatModel, spec_prompt: str) -> str:
    a = ChatAgent(model=model, system_prompt="")
    a.add_counterpart(spec_prompt)
    return a.step().strip()


@dataclass
class CamelRolePlay:
    assistant_agent: AssistantAgent
    user_agent: UserAgent
    specified_task: str
    term: TerminationConfig
    turns: list[Turn] = field(default_factory=list)

    @staticmethod
    def create_ai_society(model: ChatModel, cfg: AiSocietyConfig, term: TerminationConfig | None = None) -> "CamelRolePlay":
        term = term or TerminationConfig()

        spec_prompt = fill_angle_brackets(
            AI_SOCIETY_TASK_SPECIFIER_PROMPT,
            {
                "ASSISTANT_ROLE": cfg.assistant_role,
                "USER_ROLE": cfg.user_role,
                "TASK": cfg.preliminary_task,
                "WORD_LIMIT": str(cfg.word_limit),
            },
        )
        specified = _specify_task(model, spec_prompt)

        assistant_sys = fill_angle_brackets(
            AI_SOCIETY_ASSISTANT_SYSTEM_PROMPT,
            {"ASSISTANT_ROLE": cfg.assistant_role, "USER_ROLE": cfg.user_role, "TASK": specified},
        )
        user_sys = fill_angle_brackets(
            AI_SOCIETY_USER_SYSTEM_PROMPT,
            {"ASSISTANT_ROLE": cfg.assistant_role, "USER_ROLE": cfg.user_role, "TASK": specified},
        )

        return CamelRolePlay(
            assistant_agent=AssistantAgent(model=model, system_prompt=assistant_sys),
            user_agent=UserAgent(model=model, system_prompt=user_sys),
            specified_task=specified,
            term=term,
        )

    @staticmethod
    def create_code(model: ChatModel, cfg: CodeConfig, term: TerminationConfig | None = None) -> "CamelRolePlay":
        term = term or TerminationConfig()

        spec_prompt = fill_angle_brackets(
            CODE_TASK_SPECIFIER_PROMPT,
            {
                "DOMAIN": cfg.domain,
                "LANGUAGE": cfg.language,
                "TASK": cfg.preliminary_task,
                "WORD_LIMIT": str(cfg.word_limit),
            },
        )
        specified = _specify_task(model, spec_prompt)

        assistant_sys = fill_angle_brackets(
            CODE_ASSISTANT_SYSTEM_PROMPT,
            {"DOMAIN": cfg.domain, "LANGUAGE": cfg.language, "TASK": specified},
        )
        user_sys = fill_angle_brackets(
            CODE_USER_SYSTEM_PROMPT,
            {"DOMAIN": cfg.domain, "LANGUAGE": cfg.language, "TASK": specified},
        )

        return CamelRolePlay(
            assistant_agent=AssistantAgent(model=model, system_prompt=assistant_sys),
            user_agent=UserAgent(model=model, system_prompt=user_sys),
            specified_task=specified,
            term=term,
        )

    def _hit_token_limit(self) -> bool:
        if self.term.token_limit is None or self.term.token_model is None:
            return False
        a = count_tokens_tiktoken(self.term.token_model, self.assistant_agent.history)
        u = count_tokens_tiktoken(self.term.token_model, self.user_agent.history)
        return a >= self.term.token_limit or u >= self.term.token_limit

    def run(self, on_step: Callable[[str, str], None] | None = None) -> list[Turn]:
        """
        Paper-style termination:
          - max messages
          - user no-instruct for 3 rounds
          - user emits <CAMEL_TASK_DONE>
          - assistant starts instructing (role reversal)
          - optional token limit
        """
        user_no_instruct = 0
        msg_count = 0

        while True:
            if msg_count >= self.term.max_messages or self._hit_token_limit():
                break

            user_text = self.user_agent.step().strip()
            msg_count += 1

            if is_task_done(user_text):
                break

            if msg_count >= self.term.max_messages or self._hit_token_limit():
                break

            if parse_instruction_relaxed(user_text) is None:
                user_no_instruct += 1
            else:
                user_no_instruct = 0

            if user_no_instruct >= self.term.user_no_instruct_rounds:
                break

            self.assistant_agent.add_counterpart(user_text)
            assistant_text = self.assistant_agent.step().strip()
            msg_count += 1

            if assistant_is_instructing(assistant_text):
                break

            self.user_agent.add_counterpart(assistant_text)
            self.turns.append(Turn(user=user_text, assistant=assistant_text))

            if on_step:
                on_step(user_text, assistant_text)

        return self.turns
