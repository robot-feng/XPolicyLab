"""Ordered, observation-only demonstration frames for native KnowAct."""


def frame_indices(length, limit):
    """Evenly sample the full clip, including both ends when possible."""
    count = min(length, limit)
    return [i * (length - 1) // max(count - 1, 1) for i in range(count)]


def state_groups(length, image_limit, state_limit):
    """Group timed joint observations with the visual turn that ends each interval."""
    if state_limit < image_limit:
        raise ValueError("Joint sampling must include at least the visual anchors")
    ends = frame_indices(length, image_limit)
    indices = sorted(set(ends).union(frame_indices(length, state_limit)))
    return {end: [t for t in indices if prev < t <= end] for prev, end in zip([-1] + ends, ends)}
