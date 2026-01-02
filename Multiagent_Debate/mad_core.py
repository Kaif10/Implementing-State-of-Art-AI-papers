# mad_core.py
from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import Callable, Dict, List, Optional, Literal, Tuple, Any

from openai import OpenAI

TaskName = Literal["arithmetic", "gsm8k", "mmlu", "biography", "chess"]
OtherContextMode = Literal["concat", "summarize"]


# -----------------------------
# Paper prompts (Table 15)
# -----------------------------
def prompt_arithmetic_start(expr: str) -> str:
    return (
        f"What is the result of {expr}? "
        "Make sure to state your answer at the end of the response."
    )


def prompt_arithmetic_debate(other: str) -> str:
    return (
        "These are the recent/updated opinions from other agents: "
        f"{other} "
        "Use these opinions carefully as additional advice, can you provide an updated answer? "
        "Make sure to state your answer at the end of the response."
    )


def prompt_gsm8k_start(problem: str) -> str:
    return (
        "Can you solve the following math problem? "
        f"{problem} "
        "Explain your reasoning. Your final answer should be a single numerical number, "
        "in the form \\boxed{answer}, at the end of your response."
    )


def prompt_gsm8k_debate(other: str, problem: str) -> str:
    return (
        "These are the solutions to the problem from other agents: "
        f"{other} "
        "Using the solutions from other agents as additional information, can you provide your answer "
        "to the math problem? The original math problem is "
        f"{problem}. "
        "Your final answer should be a single numerical number, in the form \\boxed{answer}, "
        "at the end of your response."
    )


def prompt_mmlu_start(question_with_options: str) -> str:
    return (
        "Can you answer the following question as accurately as possible? "
        f"{question_with_options} "
        "Explain your answer, putting the answer in the form (X) at the end of your response."
    )


def prompt_mmlu_debate(other: str) -> str:
    return (
        "These are the solutions to the problem from other agents: "
        f"{other} "
        "Using the reasoning from other agents as additional advice, can you give an updated answer? "
        "Examine your solution and that other agents. "
        "Put your answer in the form (X) at the end of your response."
    )


def prompt_bio_start(person: str) -> str:
    return (
        f"Give a bullet point biography of {person} highlighting their contributions and achievements "
        "as a computer scientist, with each fact separated with a new line character."
    )


def prompt_bio_debate(other: str, person: str) -> str:
    return (
        f"Here are some bullet point biographies of {person} given by other agents: "
        f"{other} "
        "Closely examine your biography and the biography of other agents and provide an updated "
        "bullet point biography."
    )


def prompt_chess_start(moves_pgn: str) -> str:
    return (
        f"Here is the current sequence of moves in a chess game: {moves_pgn}. "
        "What is the best chess move I should execute next? Give a single move suggestion of the form "
        "14. <XXX> and make sure the chess move is valid in the current board state."
    )


def prompt_chess_debate(other: str) -> str:
    return (
        "Here are other chess move suggestions from other agents: "
        f"{other} "
        "Using the chess suggestions from other agents as additional advice and your earlier generated solution, "
        "can you give me your updated thoughts on the best next chess move I should play given the chess sequence ? "
        "Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state."
    )


# -----------------------------
# Answer parsing (for majority vote / early stop)
# -----------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_MMLU_RE = re.compile(r"\(([A-D])\)", re.IGNORECASE)
_SQUARE_RE = re.compile(r"\(([a-h][1-8])\)", re.IGNORECASE)  # used in chess validity tasks (not move prediction)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def parse_final(task: TaskName, text: str) -> Optional[str]:
    t = (text or "").strip()

    if task == "gsm8k":
        m = _BOXED_RE.findall(t)
        if m:
            return m[-1].strip()
        nums = _NUM_RE.findall(t)
        return nums[-1].strip() if nums else None

    if task == "mmlu":
        m = _MMLU_RE.findall(t)
        return m[-1].upper() if m else None

    if task == "arithmetic":
        nums = _NUM_RE.findall(t)
        return nums[-1].strip() if nums else None

    if task == "chess":
        # very light parsing: take last line and strip
        # (paper move format: "14. <XXX>")
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return lines[-1] if lines else None

    # biography is not “comparable” via parsing
    return None


def all_agree(parsed: List[Optional[str]]) -> bool:
    vals = [p for p in parsed if p is not None]
    return len(vals) > 0 and len(set(vals)) == 1


def majority_vote(parsed: List[Optional[str]]) -> Optional[str]:
    vals = [p for p in parsed if p is not None]
    if not vals:
        return None
    c = Counter(vals)
    # deterministic tie-break: (count desc, value asc)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# -----------------------------
# TaskSpec
# -----------------------------
@dataclass(frozen=True)
class TaskSpec:
    name: TaskName
    comparable: bool  # whether majority vote makes sense
    start_prompt: Callable[[str], str]
    debate_prompt: Callable[[str, str], str]  # (other, original_input) -> prompt


def get_task(task: TaskName) -> TaskSpec:
    if task == "arithmetic":
        return TaskSpec(
            name="arithmetic",
            comparable=True,
            start_prompt=prompt_arithmetic_start,
            debate_prompt=lambda other, _inp: prompt_arithmetic_debate(other),
        )
    if task == "gsm8k":
        return TaskSpec(
            name="gsm8k",
            comparable=True,
            start_prompt=prompt_gsm8k_start,
            debate_prompt=lambda other, inp: prompt_gsm8k_debate(other, inp),
        )
    if task == "mmlu":
        return TaskSpec(
            name="mmlu",
            comparable=True,
            start_prompt=prompt_mmlu_start,
            debate_prompt=lambda other, _inp: prompt_mmlu_debate(other),
        )
    if task == "biography":
        return TaskSpec(
            name="biography",
            comparable=False,
            start_prompt=prompt_bio_start,
            debate_prompt=lambda other, inp: prompt_bio_debate(other, inp),
        )
    if task == "chess":
        return TaskSpec(
            name="chess",
            comparable=False,  # move strings not reliably comparable for majority vote
            start_prompt=prompt_chess_start,
            debate_prompt=lambda other, _inp: prompt_chess_debate(other),
        )
    raise ValueError(f"Unknown task: {task}")


# -----------------------------
# Summarization (optional, paper Figure 13)
# -----------------------------
def summarize_other_responses(
    client: OpenAI,
    model: str,
    other_blob: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> str:
    # Minimal summarizer; used only when mode="summarize".
    messages = [
        {"role": "system", "content": "You summarize other agents' responses compactly, preserving key claims and final answers."},
        {"role": "user", "content": "Summarize the following other-agent responses into short bullet points:\n\n" + other_blob},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Core multi-agent debate
# rounds = TOTAL rounds including the initial round (paper shows Round 1, Round 2, ...)
# -----------------------------
@dataclass
class DebateConfig:
    model: str = "gpt-4o-mini"
    agents: int = 3
    rounds: int = 2              # total rounds including Round 1
    temperature: float = 0.0
    max_tokens: int = 800
    other_mode: OtherContextMode = "concat"


@dataclass
class DebateResult:
    task: TaskName
    final_text: str
    final_parsed: Optional[str]
    agent_texts: List[str]
    agent_parsed: List[Optional[str]]


def _format_other(agent_texts: List[str], exclude_i: int) -> str:
    parts: List[str] = []
    for j, t in enumerate(agent_texts):
        if j == exclude_i:
            continue
        parts.append(f"--- Agent {j+1} ---\n{t}".strip())
    return "\n\n".join(parts).strip()


def _chat(client: OpenAI, cfg: DebateConfig, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def run_debate(
    *,
    task: TaskName,
    user_input: str,
    cfg: DebateConfig,
    client: Optional[OpenAI] = None,
) -> DebateResult:
    client = client or OpenAI()
    spec = get_task(task)

    # Per-agent full conversation history (paper-style)
    histories: List[List[Dict[str, str]]] = [
        [{
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You are one of multiple independent agents solving the same problem. "
                "Think independently and do not simply mirror other agents."
            ),
        }]
        for _ in range(cfg.agents)
    ]

    # Round 1: independent answers
    start = spec.start_prompt(user_input)
    agent_texts: List[str] = []
    for i in range(cfg.agents):
        histories[i].append({"role": "user", "content": start})
        a = _chat(client, cfg, histories[i])
        histories[i].append({"role": "assistant", "content": a})
        agent_texts.append(a)

    # Rounds 2..cfg.rounds: debate updates (synchronous snapshot)
    for _round in range(2, max(cfg.rounds, 1) + 1):
        prev = agent_texts[:]  # snapshot others from previous round
        updated: List[str] = []

        for i in range(cfg.agents):
            other_blob = _format_other(prev, exclude_i=i)

            if cfg.other_mode == "summarize":
                other_blob = summarize_other_responses(client, cfg.model, other_blob, temperature=0.0, max_tokens=400)

            debate_msg = spec.debate_prompt(other_blob, user_input)
            histories[i].append({"role": "user", "content": debate_msg})
            a = _chat(client, cfg, histories[i])
            histories[i].append({"role": "assistant", "content": a})
            updated.append(a)

        agent_texts = updated

        # Early stop: if comparable task and everyone agrees (paper discusses consensus)
        if spec.comparable:
            parsed_now = [parse_final(task, t) for t in agent_texts]
            if None not in parsed_now and all_agree(parsed_now):
                break

    agent_parsed = [parse_final(task, t) for t in agent_texts]

    if spec.comparable:
        winner = majority_vote(agent_parsed)
        if winner is None:
            # fallback: return first agent response if parsing fails
            return DebateResult(task=task, final_text=agent_texts[0], final_parsed=None,
                               agent_texts=agent_texts, agent_parsed=agent_parsed)
        # Return a full agent response that matches the winning parsed answer (end-to-end faithful output).
        for txt, p in zip(agent_texts, agent_parsed):
            if p == winner:
                return DebateResult(task=task, final_text=txt, final_parsed=winner,
                                   agent_texts=agent_texts, agent_parsed=agent_parsed)
        # If parsing matched but no text aligns (should be rare), fall back safely.
        return DebateResult(task=task, final_text=agent_texts[0], final_parsed=winner,
                           agent_texts=agent_texts, agent_parsed=agent_parsed)

    # Non-comparable outputs: paper doesn’t define a majority-vote merge here
    return DebateResult(task=task, final_text=agent_texts[0], final_parsed=None,
                        agent_texts=agent_texts, agent_parsed=agent_parsed)
