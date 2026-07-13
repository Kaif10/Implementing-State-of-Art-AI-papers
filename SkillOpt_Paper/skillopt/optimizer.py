"""The SkillOpt training loop.

This is the algorithm from the paper, in plain terms. Each step:

  1. ROLL OUT   -- run the frozen agent on a batch of training questions with
                   the current skill, and score each answer.
  2. REFLECT    -- show the optimizer the failures (and some successes) and ask
                   for a few bounded edits to the skill document.
  3. EDIT       -- apply at most `learning_rate` edits to a COPY of the skill.
  4. GATE       -- score the edited copy on a held-out validation set; keep it
                   ONLY if it strictly beats the current skill's validation
                   score. Otherwise discard it and remember it as a bad edit.

Repeated over several epochs, the skill steadily accumulates guidance that
helps without overfitting -- because every change has to pass the gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .backends import Backend
from .config import SkillOptConfig
from .skill import SkillDocument
from .task import Example


# JSON shape we ask the optimizer to return.
_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "delete", "replace"]},
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["action", "index", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}


@dataclass
class StepRecord:
    epoch: int
    step: int
    train_score: float
    candidate_val_score: float
    accepted: bool
    num_edits: int


@dataclass
class TrainingResult:
    best_skill: SkillDocument
    best_val_score: float
    history: list[StepRecord] = field(default_factory=list)


class SkillOpt:
    def __init__(self, backend: Backend, task, config: SkillOptConfig | None = None):
        self.backend = backend
        self.task = task
        self.config = config or SkillOptConfig()

    # --- scoring helpers -------------------------------------------------- #
    def _score(self, skill: SkillDocument, examples: list[Example]) -> float:
        """Fraction of examples the agent gets right with this skill."""
        if not examples:
            return 0.0
        skill_text = skill.render()
        correct = 0
        for ex in examples:
            predicted = self.backend.answer(skill_text, ex.question)
            if self.task.grade(predicted, ex.answer):
                correct += 1
        return correct / len(examples)

    def _rollout(self, skill: SkillDocument, batch: list[Example]):
        """Run the agent on a batch, returning (successes, failures)."""
        skill_text = skill.render()
        successes, failures = [], []
        for ex in batch:
            predicted = self.backend.answer(skill_text, ex.question)
            row = (ex, predicted)
            if self.task.grade(predicted, ex.answer):
                successes.append(row)
            else:
                failures.append(row)
        return successes, failures

    # --- reflection prompt ------------------------------------------------ #
    def _reflection_prompt(self, skill, failures, successes, rejected, meta_notes) -> str:
        k = self.config.reflect_examples
        fail_lines = "\n".join(
            f'- question: "{ex.question}"  got: "{pred}"  correct: "{ex.answer}"'
            for ex, pred in failures[:k]
        ) or "(none)"
        ok_lines = "\n".join(
            f'- question: "{ex.question}"  got: "{pred}" (correct)'
            for ex, pred in successes[:k]
        ) or "(none)"
        rejected_lines = "\n".join(f"- {r}" for r in rejected) or "(none)"
        meta_lines = "\n".join(f"- {m}" for m in meta_notes) or "(none)"

        return f"""You are improving a SKILL document used by a fixed model to answer questions.
The skill is a numbered list of short guidance lines. Improve it so the model
fixes the failures below WITHOUT breaking the cases it already gets right.

Propose only a few precise edits (add / delete / replace). Use the line indices
shown. Prefer adding a clear, general rule over patching one example. Do not
restate guidance the skill already contains.

CURRENT SKILL:
{skill.numbered()}

FAILURES (the model's answer vs the correct answer):
{fail_lines}

CURRENTLY CORRECT (do not break these):
{ok_lines}

EDITS THAT WERE ALREADY TRIED AND DID NOT HELP (do not repeat):
{rejected_lines}

NOTES FROM EARLIER ACCEPTED EDITS:
{meta_lines}

Return JSON with an "operations" list. Each operation is
{{"action": "add"|"delete"|"replace", "index": <line number>, "text": "<line text>"}}.
For "add", index is ignored. For "delete", text is ignored."""

    @staticmethod
    def _summarize(operations: list[dict]) -> str:
        parts = []
        for op in operations:
            action = op.get("action")
            if action == "delete":
                parts.append(f"delete line {op.get('index')}")
            else:
                parts.append(f'{action}: "{(op.get("text") or "").strip()}"')
        return "; ".join(parts) or "(no-op)"

    # --- main loop -------------------------------------------------------- #
    def train(
        self,
        train_examples: list[Example],
        val_examples: list[Example],
        initial_skill: SkillDocument | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        cfg = self.config
        rng = random.Random(cfg.seed)

        working = initial_skill.copy() if initial_skill else SkillDocument()
        working_val = self._score(working, val_examples)
        best = working.copy()
        best_val = working_val

        rejected: list[str] = []
        meta_notes: list[str] = []
        history: list[StepRecord] = []

        if verbose:
            print(f"start: validation accuracy = {working_val:.3f}")

        for epoch in range(cfg.epochs):
            order = list(train_examples)
            rng.shuffle(order)
            batches = [
                order[i : i + cfg.batch_size]
                for i in range(0, len(order), cfg.batch_size)
            ]

            for step, batch in enumerate(batches):
                successes, failures = self._rollout(working, batch)
                train_score = len(successes) / max(1, len(batch))

                # Nothing to fix in this batch.
                if not failures:
                    history.append(StepRecord(epoch, step, train_score, working_val, False, 0))
                    continue

                proposal = self.backend.propose_edits(
                    self._reflection_prompt(working, failures, successes, rejected, meta_notes),
                    _EDIT_SCHEMA,
                )
                operations = proposal.get("operations", []) if isinstance(proposal, dict) else []
                candidate = working.apply(operations, cfg.learning_rate)

                # The optimizer proposed nothing usable.
                if candidate == working:
                    history.append(StepRecord(epoch, step, train_score, working_val, False, 0))
                    continue

                candidate_val = self._score(candidate, val_examples)
                num_edits = min(len(operations), cfg.learning_rate)

                # The validation gate: strictly better, or it doesn't ship.
                accepted = candidate_val > working_val
                if accepted:
                    working = candidate
                    working_val = candidate_val
                    meta_notes.append(self._summarize(operations))
                    if working_val > best_val:
                        best, best_val = working.copy(), working_val
                else:
                    rejected.append(self._summarize(operations))
                    rejected[:] = rejected[-cfg.rejected_buffer_size :]

                history.append(
                    StepRecord(epoch, step, train_score, candidate_val, accepted, num_edits)
                )

                if verbose:
                    mark = "ACCEPT" if accepted else "reject"
                    print(
                        f"epoch {epoch} step {step}: train={train_score:.2f} "
                        f"candidate_val={candidate_val:.3f} -> {mark} "
                        f"(working_val={working_val:.3f})"
                    )

        if verbose:
            print(f"done: best validation accuracy = {best_val:.3f}")

        return TrainingResult(best_skill=best, best_val_score=best_val, history=history)
