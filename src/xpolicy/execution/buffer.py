"""One owner of predicted chunks, aligned using caller-defined control ticks.

This is record-only scheduling. It neither provides a device command envelope
nor authorizes motion; ActionChunk v1 lacks a device generation/TTL guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from uuid import uuid4

import numpy as np

from xpolicy.core.config import ExecutionConfig
from xpolicy.core.contracts import ActionChunk, ActionType, ContractValidator, ModelManifest, Observation


def tick(value):
    if type(value) is not int or value < 0:
        raise ValueError("control tick must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class InferenceTicket:
    episode_id: str
    generation: int
    request_id: str
    chunk_id: str
    sequence_id: int
    source_frame_id: int
    source_clock_domain: str
    start_control_tick: int

    def context(self):
        return {"episode_id": self.episode_id, "episode_generation": self.generation,
                "request_id": self.request_id, "chunk_id": self.chunk_id,
                "sequence_id": self.sequence_id, "frame_id": self.source_frame_id,
                "start_control_tick": self.start_control_tick, "source_row": self.start_control_tick}


@dataclass(frozen=True)
class Contribution:
    ticket: InferenceTicket
    predicted_step_index: int
    weight: float

    def wire(self):
        return {**self.ticket.context(), "predicted_step_index": self.predicted_step_index, "weight": self.weight}


@dataclass(frozen=True)
class ScheduledAction:
    values: np.ndarray
    ticket: InferenceTicket
    control_tick: int
    action_seq: int
    contributors: tuple[Contribution, ...]
    valid_until_s: float | None
    clock_domain: str

    def __post_init__(self):
        values = np.array(self.values, copy=True)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("scheduled actions require a finite [D] array")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


class ActionBuffer:
    def __init__(self, manifest: ModelManifest, settings: ExecutionConfig, *, emit=None):
        if settings.mode != "record_only":
            raise ValueError("ActionBuffer currently supports record_only scheduling only")
        if settings.execute_horizon > manifest.predicted_horizon:
            raise ValueError("execute_horizon exceeds the prediction horizon")
        if settings.overlap_blend_steps and manifest.action_type != ActionType.JOINT_POSITION:
            raise ValueError("overlap blending requires absolute joint_position actions")
        self.manifest, self.settings, self.emit = manifest, settings, emit
        self.generation, self.episode_id = 0, None
        self.current_tick, self._last_taken_tick = 0, -1
        self._request_sequence, self._accepted_sequence, self._action_sequence = 0, -1, 0
        self._issued, self._pending = {}, {}

    def event(self, stage, ticket=None, **payload):
        if self.emit is not None:
            self.emit({"stage": stage, "control_tick": self.current_tick,
                       **(ticket.context() if ticket is not None else {"episode_id": self.episode_id, "episode_generation": self.generation}), **payload})

    @property
    def pending_count(self):
        return len(self._pending)

    def reset(self, episode_id: str, *, start_tick=0):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        start_tick = tick(start_tick)
        self.generation += 1
        self.episode_id = episode_id
        self.current_tick = tick(start_tick)
        self._last_taken_tick = start_tick - 1
        self._issued.clear()
        self._pending.clear()
        self._request_sequence, self._accepted_sequence, self._action_sequence = 0, -1, 0
        self.event("execution_reset", dropped_old_actions=True)
        return self.generation

    def clear(self, reason):
        count = len(self._pending)
        self._pending.clear()
        self._issued.clear()
        self.event("execution_clear", reason=reason, discarded_actions=count)

    def advance(self, control_tick):
        control_tick = tick(control_tick)
        if control_tick < self.current_tick:
            raise ValueError("control ticks cannot move backward within a generation")
        self.current_tick = control_tick
        stale = [value for value in self._pending if value < control_tick]
        for value in stale:
            del self._pending[value]
        if stale:
            self.event("execution_tick_gap", discarded_ticks=sorted(stale))

    def issue(self, obs: Observation, *, control_tick: int):
        if obs.episode_id != self.episode_id:
            raise ValueError("observation episode differs from the active execution generation")
        control_tick = tick(control_tick)
        self.advance(max(self.current_tick, control_tick))
        request = InferenceTicket(obs.episode_id, self.generation, uuid4().hex, uuid4().hex,
                                  self._request_sequence, obs.frame_id, obs.clock_domain, control_tick)
        self._request_sequence += 1
        self._issued[request.request_id] = request
        self.event("inference_request", request, model_dispatch_control_tick=self.current_tick,
                   queue_delay_ticks=self.current_tick - request.start_control_tick)
        return request

    @staticmethod
    def deadline_reason(deadline, expected_clock, now_s, clock_domain):
        if deadline is None:
            return None
        if (clock_domain != expected_clock or isinstance(now_s, bool) or
                not isinstance(now_s, Real) or not isfinite(now_s)):
            return "unverified_deadline_clock"
        return "expired_deadline" if now_s >= deadline else None

    def accept(self, request: InferenceTicket, chunk: ActionChunk, *, control_tick: int,
               now_s=None, clock_domain=None):
        def discard(reason):
            self.event("prediction_discard", request, reason=reason)
            return False
        # Context is checked before parsing an obsolete response: reset cannot
        # be poisoned by errors or malformed results from a previous episode.
        if request.generation != self.generation or request.episode_id != self.episode_id:
            return discard("old_generation")
        if self._issued.get(request.request_id) != request:
            return discard("duplicate_or_unknown_request")
        del self._issued[request.request_id]
        if request.sequence_id <= self._accepted_sequence:
            return discard("out_of_order_result")
        self.advance(control_tick)
        ContractValidator.prediction(chunk, self.manifest, source_frame_id=request.source_frame_id,
                                     episode_id=request.episode_id, sequence_id=request.sequence_id,
                                     clock_domain=request.source_clock_domain)
        reason = self.deadline_reason(chunk.valid_until_s, chunk.clock_domain, now_s, clock_domain)
        if reason:
            return discard(reason)
        first_tick = max(self.current_tick, self._last_taken_tick + 1)
        trim = max(0, first_tick - request.start_control_tick) if self.settings.latency_trim else 0
        stop = min(chunk.horizon, trim + self.settings.execute_horizon)
        if trim >= stop:
            return discard("prediction_horizon_elapsed")
        values = chunk.values[trim:stop]
        self.event("execution_trim", request, trace_layer="A1", trimmed_steps=trim,
                   predicted_indices=list(range(trim, stop)), first_control_tick=first_tick,
                   values=values.tolist(), latency_trim=self.settings.latency_trim)
        overlap = 0
        for offset in range(min(self.settings.overlap_blend_steps, len(values))):
            if first_tick + offset not in self._pending:
                break
            old = self._pending[first_tick + offset]
            if self.deadline_reason(old[2], old[3], now_s, clock_domain) is not None:
                break
            overlap += 1
        # Replacement is latest-only. The old future beyond the new horizon
        # is removed, so an older tail cannot reappear after the new plan ends.
        previous, self._pending = self._pending, {}
        for offset, value in enumerate(values):
            control = first_tick + offset
            contributions = (Contribution(request, trim + offset, 1.0),)
            deadline = chunk.valid_until_s
            if offset < overlap:
                old_value, old_sources, old_deadline, old_clock, _ = previous[control]
                old_reason = self.deadline_reason(old_deadline, old_clock, now_s, clock_domain)
                if old_reason is None:
                    alpha = (offset + 1) / (overlap + 1)
                    value = (1 - alpha) * old_value.astype(np.float64) + alpha * value.astype(np.float64)
                    contributions = tuple(Contribution(item.ticket, item.predicted_step_index, item.weight * (1 - alpha)) for item in old_sources) + (Contribution(request, trim + offset, alpha),)
                    deadlines = [item for item in (old_deadline, deadline) if item is not None]
                    deadline = min(deadlines) if deadlines else None
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            self._pending[control] = (owned, contributions, deadline, chunk.clock_domain, request)
        self._accepted_sequence = request.sequence_id
        self.event("execution_buffer", request, trace_layer="A2", first_control_tick=first_tick,
                   overlap_steps=overlap, algorithm="linear_absolute_joint_overlap_v1",
                   values=[self._pending[first_tick + offset][0].tolist() for offset in range(len(values))],
                   contributors=[[item.wire() for item in self._pending[first_tick + offset][1]] for offset in range(len(values))])
        return True

    def take(self, control_tick: int, *, now_s=None, clock_domain=None):
        self.advance(control_tick)
        if control_tick <= self._last_taken_tick:
            raise ValueError("a control tick can only be consumed once")
        self._last_taken_tick = control_tick
        pending = self._pending.pop(control_tick, None)
        if pending is None:
            self.event("execution_empty", behavior=self.settings.empty_buffer)
            return None
        values, contributors, deadline, source_clock, request = pending
        reason = self.deadline_reason(deadline, source_clock, now_s, clock_domain)
        if reason:
            self.event("execution_command_discard", request, reason=reason)
            return None
        result = ScheduledAction(values, request, control_tick, self._action_sequence, contributors, deadline, source_clock)
        self._action_sequence += 1
        self.event("command", request, trace_layer="A3", action_seq=result.action_seq,
                   predicted_step_index=contributors[-1].predicted_step_index,
                   values=values.tolist(), contributors=[item.wire() for item in contributors],
                   actuator="record_only", status="scheduled", device_feedback=None)
        return result
