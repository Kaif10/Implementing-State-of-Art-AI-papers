"""Training settings, with names that mirror ordinary deep-learning training.

The whole point of SkillOpt is that you train a skill document with the same
knobs you'd use to train weights: epochs, batch size, a learning rate, and a
validation gate. These are those knobs.
"""

from dataclasses import dataclass


@dataclass
class SkillOptConfig:
    # How the data is split.
    train_size: int = 40        # questions used to find mistakes
    val_size: int = 20          # held-out questions used to accept/reject edits
    test_size: int = 40         # held-out questions used only for the final report

    # Training schedule.
    epochs: int = 3             # passes over the training data
    batch_size: int = 8         # questions per optimization step

    # "Learning rate": the most edits (add/delete/replace lines) allowed in a
    # single step. A small budget keeps each step from rewriting the whole
    # document and destabilizing training -- exactly like a small numeric LR.
    learning_rate: int = 4

    # How many example successes/failures to show the optimizer when it
    # reflects on a batch. Keeps the reflection prompt small.
    reflect_examples: int = 6

    # Remember this many recently-rejected edits and tell the optimizer not to
    # repeat them (the paper's "rejected-edit buffer").
    rejected_buffer_size: int = 10

    seed: int = 0
