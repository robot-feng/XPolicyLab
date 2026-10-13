"""Capture planned actions without an actuator or fabricated feedback."""

import numpy as np

from xpolicy.core.contracts import ActionChunk
from xpolicy.execution.buffer import ScheduledAction


class RecordOnlySink:
    def __init__(self):
        self.actions = []
        self._last_command = None

    def submit(self, chunk: ActionChunk, count: int):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= chunk.horizon:
            raise ValueError("record count must lie within the prediction horizon")
        values = np.array(chunk.values[:count], copy=True)
        values.setflags(write=False)
        self.actions.append(values)
        return values

    def submit_commands(self, commands):
        commands = tuple(commands)
        if not commands or any(not isinstance(command, ScheduledAction) for command in commands):
            raise ValueError("record-only submission requires scheduled commands")
        previous = self._last_command
        for command in commands:
            if previous is not None:
                if (command.ticket.generation, command.ticket.episode_id) != (previous.ticket.generation, previous.ticket.episode_id):
                    raise ValueError("a record-only sink cannot mix execution generations")
                if command.control_tick <= previous.control_tick or command.action_seq <= previous.action_seq:
                    raise ValueError("duplicate or out-of-order recorded command")
            previous = command
        values = np.stack([command.values for command in commands])
        values.setflags(write=False)
        self.actions.append(values)
        self._last_command = previous
        return values
