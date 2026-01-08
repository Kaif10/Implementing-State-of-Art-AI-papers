from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from agents import AssistantAgent, ChatAgent, ChatModel, UserAgent
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
    # One back-and-forth exchange.
    user: str
    assistant: str


@dataclass(frozen=True)
class Instruction:
    # Parsed structured instruction from the user role.
    instruction: str
    input: str


def is_task_done(text: str) -> bool:
    # Exact stop token used by the user role.
    return text.strip() == CAMEL_TASK_DONE


# Regexes for the structured instruction format in the user prompt.
_INSTRUCTION_RE = re.compile(r"^\s*Instruction\s*:\s*(.+?)\s*$", re.IGNORECASE)
_INPUT_RE = re.compile(r"^\s*Input\s*:\s*(.+?)\s*$", re.IGNORECASE)


# Treat the user as "instructing" if BOTH an Instruction: line and an Input: line appear anywhere (case-insensitive). Returns the first occurrences.
   
def parse_instruction(text: str) -> Instruction | None:
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


# Termination: assistant begins issuing 'Instruction:' (role reversal). Check first non-empty line.
def assistant_is_instructing(text: str) -> bool:
    for ln in (x.strip() for x in text.splitlines()):
        if ln:
            return ln.startswith("Instruction:")
    return False


@dataclass(frozen=True)
class TerminationConfig:
    max_messages: int = 40
    user_no_instruct_rounds: int = 3


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
    # One-shot task refinement using the task specifier prompt.
    a = ChatAgent(model=model, system_prompt="")
    a.add_counterpart(spec_prompt)
    return a.step().strip()


@dataclass
class CamelRolePlay:
    # Holds role agents and the ongoing conversation state.
    assistant_agent: AssistantAgent
    user_agent: UserAgent
    specified_task: str
    term: TerminationConfig
    turns: list[Turn] = field(default_factory=list)

    @staticmethod
    def create_ai_society(model: ChatModel, cfg: AiSocietyConfig, term: TerminationConfig | None = None) -> "CamelRolePlay":
        # Build AI Society roleplay with a refined task and role-specific system prompts.
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
        # Build Code roleplay with a refined task and domain/language prompts.
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


    def run(self, on_step: Callable[[str, str], None] | None = None) -> list[Turn]:
    # Paper-style termination: - max messages, user no-instruct for 3 rounds, user emits <CAMEL_TASK_DONE>, assistant starts instructing (role reversal)        
        user_no_instruct = 0
        msg_count = 0

        while True:
            if msg_count >= self.term.max_messages:
                break

            # User speaks first in each turn.
            user_text = self.user_agent.step().strip()
            msg_count += 1

            if is_task_done(user_text):
                break

            if msg_count >= self.term.max_messages:
                break

            # Track whether the user is still giving structured instructions.
            if parse_instruction(user_text) is None:
                user_no_instruct += 1
            else:
                user_no_instruct = 0

            if user_no_instruct >= self.term.user_no_instruct_rounds:
                break

            # Assistant responds conditioned on the user message.
            self.assistant_agent.add_counterpart(user_text)
            assistant_text = self.assistant_agent.step().strip()
            msg_count += 1

            if assistant_is_instructing(assistant_text):
                break

            # Feed assistant reply back to the user agent for the next round.
            self.user_agent.add_counterpart(assistant_text)
            self.turns.append(Turn(user=user_text, assistant=assistant_text))

            if on_step:
                on_step(user_text, assistant_text)

        return self.turns
