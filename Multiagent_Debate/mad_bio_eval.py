# mad_bio_eval.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict

from openai import OpenAI


CRITIC_PROMPT = (
    "Consider the following biography of {person}: {generated_bio} "
    "Is the above biography above consistent with the fact below? "
    "{ground_truth_fact} "
    "Give a single-word answer, yes, no, or uncertain."
)


def load_facts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[-•\*\u2022]\s*", "", ln).strip()
        if ln:
            cleaned.append(ln)
    return cleaned


def _norm_label(x: str) -> str:
    t = (x or "").strip().lower()
    t = re.split(r"\s+", t)[0] if t else ""
    if t in {"yes", "no", "uncertain"}:
        return t
    # tiny robustness
    if t in {"y", "true", "consistent"}:
        return "yes"
    if t in {"n", "false", "inconsistent"}:
        return "no"
    return "uncertain"


@dataclass
class BioEvalResult:
    total_facts: int
    yes: int
    no: int
    uncertain: int
    decided: int              # yes + no
    decided_rate: float       # decided / total_facts
    yes_rate_over_decided: float  # yes / decided (if decided>0)
    labels: List[str]


def eval_biography(
    *,
    person: str,
    generated_bio: str,
    ground_truth_facts: List[str],
    critic_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8,
    client: Optional[OpenAI] = None,
) -> BioEvalResult:
    client = client or OpenAI()

    labels: List[str] = []
    yes = no = uncertain = 0

    for fact in ground_truth_facts:
        msg = CRITIC_PROMPT.format(
            person=person,
            generated_bio=generated_bio,
            ground_truth_fact=fact,
        )

        resp = client.chat.completions.create(
            model=critic_model,
            messages=[
                {"role": "system", "content": "Reply with exactly one word: yes, no, or uncertain."},
                {"role": "user", "content": msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        lab = _norm_label(resp.choices[0].message.content or "")
        labels.append(lab)

        if lab == "yes":
            yes += 1
        elif lab == "no":
            no += 1
        else:
            uncertain += 1

    total = len(labels)
    decided = yes + no
    decided_rate = (decided / total) if total else 0.0
    yes_rate_over_decided = (yes / decided) if decided else 0.0

    return BioEvalResult(
        total_facts=total,
        yes=yes,
        no=no,
        uncertain=uncertain,
        decided=decided,
        decided_rate=decided_rate,
        yes_rate_over_decided=yes_rate_over_decided,
        labels=labels,
    )
