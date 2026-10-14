"""Episode context and immutable recorded-environment receipts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    generation: int
    seed: int
    repeat_index: int

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("environment episode ID must be a nonempty string")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("environment generation must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 4294967295:
            raise ValueError("environment seed must lie within the NumPy seed range")
        if type(self.repeat_index) is not int or self.repeat_index < 0:
            raise ValueError("environment repeat index must be a nonnegative integer")


@dataclass(frozen=True)
class RecordedSubmission:
    values: np.ndarray
    labels: np.ndarray | None
    control_frame_ids: tuple[int, ...]
    control_timestamps_s: tuple[float, ...]
    causal: bool = False
    device_feedback: None = None
