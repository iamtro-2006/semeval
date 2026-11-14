"""Drive uploads happen in the launcher after successful training.

Intermediate checkpoints stay local. Keep this hook for existing callers.
"""


def checkpoint_callbacks():
    return []
