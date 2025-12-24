from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from kb_index import Chunk, FaissKB


# ---------------------------
# Schemas: SINGLE generator emits reflection labels (no critic at inference)
# ---------------------------

class RetrieveDecision(BaseModel):
    decision: str = Field(description="yes, no, or continue")
    query: str = Field(description="query if decision=yes, else empty")


class Candidate(BaseModel):
    chunk_id: str
    sentence: str
    is_final: bool
    isrel: str = Field(description="relevant or irrelevant")
    issup: str = Field(description="fully_supported, partially_supported, or no_support")
    isuse: int = Field(description="1-5 usefulness")


class CandidateBatch(BaseModel):
    candidates: List[Candidate]


class NoCtxStep(BaseModel):
    sentence: str
    is_final: bool
    isuse: int = Field(description="1-5 usefulness")


# ---------------------------
# Helpers
# ---------------------------

def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s.strip('"').strip("'")


def _format_passage(c: Chunk) -> str:
    return f'<paragraph id="{c.chunk_id}" source="{Path(c.source).name}">\n{c.text}\n</paragraph>'


def _rel_score(isrel: str) -> float:
    return 1.0 if isrel.strip().lower() == "relevant" else 0.0


def _sup_score(issup: str) -> float:
    x = issup.strip().lower()
    if x == "fully_supported":
        return 1.0
    if x == "partially_supported":
        return 0.5
    return 0.0


@dataclass
class Beam:
    draft: List[str]
    ctx: List[Chunk]
    score: float
    done: bool = False


# Self-RAG inference loop

class SelfRAG:
    """
    Faithful inference behavior:
    - Decide Retrieve per segment (sentence) => yes/no/continue
    - If retrieve: generate one candidate per passage AND emit reflection labels (IsRel/IsSup/IsUse)
    - Select by reflection scores (hard-gate irrelevant/unsupported)
    - Optional segment-level beam search (paper-style) via beam_size > 1
    """

    def __init__(
        self,
        client: OpenAI,
        kb: FaissKB,
        *,
        gen_model: str,
        top_k: int = 5,
        per_step_passages: int = 3,
        max_steps: int = 10,
        weights: Tuple[float, float, float] = (1.0, 2.0, 1.0),  # rel, sup, use
        temperature: float = 0.0,
        require_supported: bool = True,
        require_relevant: bool = True,
    ):
        self.client = client
        self.kb = kb

        self.gen_model = gen_model
        self.top_k = top_k
        self.per_step_passages = per_step_passages
        self.max_steps = max_steps
        self.w_rel, self.w_sup, self.w_use = weights
        self.temperature = temperature

        self.require_supported = require_supported
        self.require_relevant = require_relevant

    # ---------------------------
    # Generator calls
    # ---------------------------

    def decide_retrieve(self, question: str, draft: str) -> RetrieveDecision:
        # The main hallucination cause is "Retrieve=no" on doc-questions. Be conservative.
        resp = self.client.responses.parse(
            model=self.gen_model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You control retrieval for a RAG system. "
                        "Be conservative: retrieve for policies, requirements, definitions, numbers, and quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Question:
{question}

Draft so far:
{draft if draft else "(empty)"}

Decide Retrieve for the NEXT sentence.

Rules:
- Default to decision="yes".
- Use decision="no" ONLY if the next sentence is general reasoning and would still be correct if no documents existed.
- Use decision="continue" only if you already have relevant retrieved context and can reuse it.
- If decision="yes", provide a short keyword-style query that would retrieve the needed text.

Return RetrieveDecision (decision, query).""",
                },
            ],
            text_format=RetrieveDecision,
        )
        out: RetrieveDecision = resp.output_parsed
        out.decision = out.decision.strip().lower()
        if out.decision not in {"yes", "no", "continue"}:
            out.decision = "yes"
        return out

    def generate_candidates(self, question: str, draft: str, passages: List[Chunk]) -> CandidateBatch:
        passages_txt = "\n\n".join(_format_passage(p) for p in passages)

    
        resp = self.client.responses.parse(
            model=self.gen_model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Generate one candidate next sentence per passage, plus reflection labels. "
                        "Do not add facts not supported by the passage."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Question:
{question}

Draft so far (do not repeat it):
{draft if draft else "(empty)"}

Passages:
{passages_txt}

For EACH passage, return:
- chunk_id (must match passage id)
- sentence: EXACTLY ONE next sentence using ONLY that passage
- is_final: true ONLY if the question is fully answered after this sentence
- isrel: relevant/irrelevant
- issup: fully_supported/partially_supported/no_support
- isuse: 1-5 usefulness

Utility rubric (follow strictly):
- If the question asks "what counts as / what is / define" or "give examples",
  the BEST next sentence is a DEFINITION or LIST of items from the passage
  (e.g., "Restricted data includes ...").
- If the question asks for examples, include at least TWO examples from the passage in the sentence.
- A constraint-only sentence (e.g., "must not be stored...") MUST have isuse <= 2
  if it does NOT include the requested definition/examples.
- Only set is_final=true if the sentence alone completes the requested definition/list/examples.

If the passage does not contain the needed info:
- set issup="no_support"
- write a brief admission rather than guessing.

Return CandidateBatch.""",
                },
            ],
            text_format=CandidateBatch,
        )

        batch: CandidateBatch = resp.output_parsed
        for c in batch.candidates:
            c.sentence = _clean_sentence(c.sentence)
            c.isrel = c.isrel.strip().lower()
            c.issup = c.issup.strip().lower()
            c.isuse = max(1, min(5, int(c.isuse)))
        return batch

    def generate_noctx(self, question: str, draft: str) -> NoCtxStep:
        
        resp = self.client.responses.parse(
            model=self.gen_model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": "Write exactly ONE next sentence with NO retrieved passages. Never guess policy facts.",
                },
                {
                    "role": "user",
                    "content": f"""Question:
{question}

Draft so far:
{draft if draft else "(empty)"}

You have NO retrieved passages.

Rules:
- Do NOT guess or generalize policy facts (avoid 'typically', 'often', 'generally').
- If the question asks for policy/requirements/definitions/numbers that should come from documents,
  output exactly:
  sentence = "I can't answer that from the provided documents without retrieval."
  is_final = true
  isuse = 1
- Otherwise, write ONE safe next sentence and output isuse (1-5).""",
                },
            ],
            text_format=NoCtxStep,
        )
        out: NoCtxStep = resp.output_parsed
        out.sentence = _clean_sentence(out.sentence)
        out.isuse = max(1, min(5, int(out.isuse)))
        return out


    # Scoring / selection

    def _score_candidate(self, c: Candidate) -> float:
        return (
            self.w_rel * _rel_score(c.isrel)
            + self.w_sup * _sup_score(c.issup)
            + self.w_use * (c.isuse / 5.0)
        )

    def _score_noctx(self, isuse: int) -> float:
        return self.w_use * (isuse / 5.0)

    def _accept(self, c: Candidate) -> bool:
        if self.require_relevant and c.isrel != "relevant":
            return False
        if self.require_supported and c.issup == "no_support":
            return False
        return True

  
    # Public API
   

    def answer(self, question: str, *, beam_size: int = 1) -> str:
        if beam_size <= 1:
            return self._answer_greedy(question)
        return self._answer_beam(question, beam_size)
 
    # Greedy (beam=1)
  
    def _answer_greedy(self, question: str) -> str:
        draft: List[str] = []
        ctx: List[Chunk] = []

        for _ in range(self.max_steps):
            draft_txt = " ".join(draft).strip()
            dec = self.decide_retrieve(question, draft_txt)

            if dec.decision == "yes":
                ctx = self.kb.search(dec.query.strip() or question, self.top_k)
            elif dec.decision == "no":
                ctx = []
            # continue => keep ctx

            if ctx:
                passages = ctx[: self.per_step_passages]
                batch = self.generate_candidates(question, draft_txt, passages)

                best: Optional[Candidate] = None
                best_score = -1e9
                for c in batch.candidates:
                    if not c.sentence or not self._accept(c):
                        continue
                    s = self._score_candidate(c)
                    if s > best_score:
                        best_score = s
                        best = c

                if not best:
                    break

                draft.append(f"{best.sentence} [{best.chunk_id}]")
                if best.is_final:
                    break
            else:
                step = self.generate_noctx(question, draft_txt)
                if not step.sentence:
                    break
                draft.append(step.sentence)
                if step.is_final:
                    break

        return " ".join(draft).strip()

   
    # Segment-level beam search (paper-style)

    def _answer_beam(self, question: str, beam_size: int) -> str:
        beams: List[Beam] = [Beam(draft=[], ctx=[], score=0.0, done=False)]

        for _ in range(self.max_steps):
            expanded: List[Beam] = []

            for b in beams:
                if b.done:
                    expanded.append(b)
                    continue

                draft_txt = " ".join(b.draft).strip()
                dec = self.decide_retrieve(question, draft_txt)

                if dec.decision == "yes":
                    ctx = self.kb.search(dec.query.strip() or question, self.top_k)
                elif dec.decision == "no":
                    ctx = []
                else:  # continue
                    ctx = b.ctx

                if ctx:
                    passages = ctx[: self.per_step_passages]
                    batch = self.generate_candidates(question, draft_txt, passages)

                    for cand in batch.candidates:
                        if not cand.sentence or not self._accept(cand):
                            continue
                        expanded.append(
                            Beam(
                                draft=b.draft + [f"{cand.sentence} [{cand.chunk_id}]"],
                                ctx=ctx,
                                score=b.score + self._score_candidate(cand),
                                done=bool(cand.is_final),
                            )
                        )
                else:
                    step = self.generate_noctx(question, draft_txt)
                    if not step.sentence:
                        continue
                    expanded.append(
                        Beam(
                            draft=b.draft + [step.sentence],
                            ctx=[],
                            score=b.score + self._score_noctx(step.isuse),
                            done=bool(step.is_final),
                        )
                    )

            if not expanded:
                break

            # keep best unique drafts (prevents beam spam)
            uniq = {}
            for b in expanded:
                key = " ".join(b.draft)
                prev = uniq.get(key)
                if prev is None or b.score > prev.score:
                    uniq[key] = b

            beams = sorted(uniq.values(), key=lambda x: x.score, reverse=True)[:beam_size]
            if all(b.done for b in beams):
                break

        best = max(beams, key=lambda x: x.score)
        return " ".join(best.draft).strip()
