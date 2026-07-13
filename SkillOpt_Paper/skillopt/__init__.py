"""A minimal, honest implementation of Microsoft's SkillOpt.

Paper: "SkillOpt: Executive Strategy for Self-Evolving Agent Skills"
(Yang et al., Microsoft Research, arXiv:2605.23904).

Core idea: the *skill document* (a short piece of natural-language guidance)
is treated as the trainable parameter of a *frozen* language model. We never
touch the model's weights. Instead we run the model on training questions,
read its mistakes, and edit the skill document to fix them -- keeping an edit
only if it improves a held-out validation score.
"""

from .config import SkillOptConfig
from .skill import SkillDocument
from .optimizer import SkillOpt
from .task import NumberFormattingTask, Example
from .backends import AnthropicBackend, SimulatedBackend

__all__ = [
    "SkillOptConfig",
    "SkillDocument",
    "SkillOpt",
    "NumberFormattingTask",
    "Example",
    "AnthropicBackend",
    "SimulatedBackend",
]
