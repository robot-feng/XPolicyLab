"""Capture planned actions without an actuator or fabricated feedback."""

import numpy as np

from xpolicy.core.contracts import ActionChunk


class RecordOnlySink:
    def __init__(self):
        self.actions = []

    def submit(self, chunk: ActionChunk, count: int):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= chunk.horizon:
            raise ValueError("record count must lie within the prediction horizon")
        values = np.array(chunk.values[:count], copy=True)
        values.setflags(write=False)
        self.actions.append(values)
        return values
