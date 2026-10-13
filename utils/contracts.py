"""Versioned policy/runtime contracts shared by adapters and transports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np


class ActionType(str, Enum):
    """Physical interpretation of an action vector."""

    JOINT_POSITION = "joint_position"
    JOINT_DELTA = "joint_delta"
    JOINT_VELOCITY = "joint_velocity"
    EEF_POSE = "eef_pose"


def _names(value: Sequence[str] | None, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of names, not a string.")
    names = tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{field_name} entries must be non-empty strings.")
    return names


@dataclass(frozen=True)
class ActionChunk:
    """A complete policy prediction, before execution consumes any steps.

    The array is always ``[horizon, action_dim]``. Robot-specific units and
    joint names are explicit when known. Missing execution metadata stays
    missing; callers can use :attr:`execution_ready` to fail closed before
    sending this chunk to a device.
    """

    values: np.ndarray
    action_type: ActionType | str
    action_units: Sequence[str] | None = None
    source_frame_id: int | None = None
    episode_id: str | None = None
    sequence_id: int | None = None
    joint_names: Sequence[str] | None = None
    reference_frame: str | None = None
    valid_until_s: float | None = None
    clock_domain: str | None = None

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        try:
            action_type = ActionType(self.action_type)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in ActionType)
            raise ValueError(
                f"Unsupported action_type {self.action_type!r}; expected one of: {choices}."
            ) from exc
        object.__setattr__(self, "action_type", action_type)

        values = np.asarray(self.values)
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError(
                f"ActionChunk.values must have shape [horizon, action_dim] with both "
                f"dimensions non-zero, got {values.shape}."
            )
        if values.dtype.kind not in "iuf":
            raise TypeError(
                f"ActionChunk.values must contain real numeric values, got {values.dtype}."
            )
        if not np.isfinite(values).all():
            raise ValueError("ActionChunk.values must contain only finite values.")
        # Chunks are small relative to images. An owned, read-only array keeps
        # concurrent tracing and execution from observing caller mutation.
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

        units = _names(self.action_units, "action_units")
        if units is not None and len(units) != values.shape[1]:
            raise ValueError(
                f"action_units has {len(units)} entries for action_dim={values.shape[1]}."
            )
        object.__setattr__(self, "action_units", units)

        joint_names = _names(self.joint_names, "joint_names")
        if joint_names is not None:
            if len(joint_names) != values.shape[1]:
                raise ValueError(
                    f"joint_names has {len(joint_names)} entries for action_dim={values.shape[1]}."
                )
            if len(set(joint_names)) != len(joint_names):
                raise ValueError("joint_names must be unique and preserve action order.")
        object.__setattr__(self, "joint_names", joint_names)

        for field_name in ("source_frame_id", "sequence_id"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None.")
            if value is not None:
                object.__setattr__(self, field_name, int(value))
        if self.episode_id is not None and (
            not isinstance(self.episode_id, str) or not self.episode_id.strip()
        ):
            raise ValueError("episode_id must be a non-empty string or None.")
        if self.reference_frame is not None and (
            not isinstance(self.reference_frame, str) or not self.reference_frame.strip()
        ):
            raise ValueError("reference_frame must be a non-empty string or None.")
        if self.clock_domain is not None and (
            not isinstance(self.clock_domain, str) or not self.clock_domain.strip()
        ):
            raise ValueError("clock_domain must be a non-empty string or None.")
        if self.valid_until_s is not None and (
            isinstance(self.valid_until_s, bool)
            or not isinstance(self.valid_until_s, Real)
            or not isfinite(self.valid_until_s)
        ):
            raise ValueError("valid_until_s must be a finite timestamp or None.")
        if self.valid_until_s is not None:
            object.__setattr__(self, "valid_until_s", float(self.valid_until_s))

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def execution_gaps(self) -> tuple[str, ...]:
        """Fields a strict execution path still needs before it may consume this chunk."""
        missing = [
            name
            for name in (
                "action_units",
                "source_frame_id",
                "episode_id",
                "sequence_id",
                "valid_until_s",
                "clock_domain",
            )
            if getattr(self, name) is None
        ]
        if self.action_type in {
            ActionType.JOINT_POSITION,
            ActionType.JOINT_DELTA,
            ActionType.JOINT_VELOCITY,
        }:
            if self.joint_names is None:
                missing.append("joint_names")
        elif self.reference_frame is None:
            missing.append("reference_frame")
        return tuple(missing)

    @property
    def execution_ready(self) -> bool:
        """Metadata completeness only; device bounds, expiry and permission need local checks."""
        return not self.execution_gaps

    def require_execution_ready(self) -> ActionChunk:
        """Raise with the exact missing contract fields before device execution."""
        missing = self.execution_gaps
        if missing:
            raise ValueError(
                "ActionChunk is not ready for strict execution; missing: "
                + ", ".join(missing)
            )
        return self

    def to_wire_dict(self) -> dict[str, Any]:
        """Return a msgpack-compatible, explicitly versioned representation."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "values": self.values,
            "action_type": self.action_type.value,
            "action_units": (
                list(self.action_units) if self.action_units is not None else None
            ),
            "source_frame_id": self.source_frame_id,
            "episode_id": self.episode_id,
            "sequence_id": self.sequence_id,
            "joint_names": list(self.joint_names) if self.joint_names is not None else None,
            "reference_frame": self.reference_frame,
            "valid_until_s": self.valid_until_s,
            "clock_domain": self.clock_domain,
            "execution_ready": self.execution_ready,
        }

    @classmethod
    def from_wire_dict(cls, data: Mapping[str, Any]) -> ActionChunk:
        if not isinstance(data, Mapping):
            raise TypeError("ActionChunk wire value must be a mapping.")
        version = data.get("schema_version")
        if type(version) is not int or version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ActionChunk schema_version {version!r}; "
                f"expected {cls.SCHEMA_VERSION}."
            )
        return cls(
            values=data.get("values"),
            action_type=data.get("action_type"),
            action_units=data.get("action_units"),
            source_frame_id=data.get("source_frame_id"),
            episode_id=data.get("episode_id"),
            sequence_id=data.get("sequence_id"),
            joint_names=data.get("joint_names"),
            reference_frame=data.get("reference_frame"),
            valid_until_s=data.get("valid_until_s"),
            clock_domain=data.get("clock_domain"),
        )
