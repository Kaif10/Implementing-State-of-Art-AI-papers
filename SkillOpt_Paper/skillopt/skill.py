"""The skill document -- the thing we are training.

We represent it as an ordered list of short guidance lines. Editing it means
adding, deleting, or replacing lines. Keeping it as a list (rather than free
text) is what makes "bounded edits" well-defined: one operation touches one
line, so we can count edits and cap them with the learning rate.

When deployed, the document is just rendered to a Markdown file
(`best_skill.md`) and handed to the frozen model unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillDocument:
    lines: list[str] = field(default_factory=list)

    def copy(self) -> "SkillDocument":
        return SkillDocument(list(self.lines))

    def render(self) -> str:
        """Markdown shown to the model (and saved as best_skill.md)."""
        if not self.lines:
            return "# Skill\n\n(empty)\n"
        body = "\n".join(f"- {line}" for line in self.lines)
        return f"# Skill\n\n{body}\n"

    def numbered(self) -> str:
        """Lines with 0-based indices, so the optimizer can target them."""
        if not self.lines:
            return "(the skill is currently empty)"
        return "\n".join(f"[{i}] {line}" for i, line in enumerate(self.lines))

    def apply(self, operations: list[dict], max_edits: int) -> "SkillDocument":
        """Return a new document with up to `max_edits` operations applied.

        Each operation is a dict: {"action": "add"|"delete"|"replace",
        "index": int, "text": str}. Invalid operations are skipped rather than
        raising -- the optimizer's output is untrusted, so we stay robust.

        Deletes and replaces are resolved against the *original* indices and
        applied together, so several edits in one step don't shift each other's
        targets.
        """
        ops = operations[:max_edits]
        n = len(self.lines)

        replace: dict[int, str] = {}
        delete: set[int] = set()
        additions: list[str] = []

        for op in ops:
            action = (op.get("action") or "").strip().lower()
            text = (op.get("text") or "").strip()
            try:
                index = int(op.get("index", -1))
            except (TypeError, ValueError):
                index = -1

            if action == "add":
                if text:
                    additions.append(text)
            elif action == "delete":
                if 0 <= index < n:
                    delete.add(index)
            elif action == "replace":
                if 0 <= index < n and text:
                    replace[index] = text

        new_lines: list[str] = []
        for i, line in enumerate(self.lines):
            if i in delete:
                continue
            new_lines.append(replace.get(i, line))
        new_lines.extend(additions)

        return SkillDocument(new_lines)

    def __eq__(self, other) -> bool:
        return isinstance(other, SkillDocument) and self.lines == other.lines

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.render())

    @classmethod
    def load(cls, path: str) -> "SkillDocument":
        lines: list[str] = []
        with open(path) as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if raw.startswith("- "):
                    lines.append(raw[2:].strip())
        return cls(lines)
