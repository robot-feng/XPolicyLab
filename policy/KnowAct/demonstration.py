"""Ordered, observation-only demonstration frames for native KnowAct."""


def frame_indices(length, limit):
    """Evenly sample the full clip, including both ends when possible."""
    count = min(length, limit)
    return [i * (length - 1) // max(count - 1, 1) for i in range(count)]
