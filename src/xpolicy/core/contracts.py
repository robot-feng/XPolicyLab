"""Explicit training, dataset and robot boundary semantics.

ActionChunk has one implementation in XPolicyLab.utils.contracts during the
package migration. Importing core never imports a policy, simulator or SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator,
)

from XPolicyLab.utils.contracts import ActionChunk, ActionType
from xpolicy.core.errors import ContractError, ContractIssue

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Rate = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_is_integer(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class ArtifactIdentity(Schema):
    uri: Name
    sha256: Sha256


class FieldSpec(Schema):
    name: Name
    unit: Name
    reference_frame: Name | None = None

    @field_validator("name", "unit")
    @classmethod
    def no_placeholder(cls, value):
        if value.lower() in {"auto", "unknown", "normalized"} or "REQUIRED_" in value:
            raise ValueError("physical fields must be declared explicitly")
        return value


class CameraSpec(Schema):
    name: Name
    color_space: Literal["RGB"] = "RGB"
    # Boundary dimensions, not policy-specific tensor resize/crop dimensions.
    shape: tuple[Annotated[int, Field(gt=0, strict=True)], Annotated[int, Field(gt=0, strict=True)], Literal[3]] | None = None
    source_key: Name | None = None


class BoundaryProfile(Schema):
    cameras: tuple[CameraSpec, ...]
    state_fields: tuple[FieldSpec, ...] = ()
    action_fields: tuple[FieldSpec, ...]
    action_type: ActionType
    reference_frame: Name | None = None
    quaternion_order: Literal["wxyz", "xyzw"] | None = None

    @model_validator(mode="after")
    def named_layouts(self):
        for name in ("cameras", "state_fields", "action_fields"):
            fields = getattr(self, name)
            names = [item.name for item in fields]
            if len(set(names)) != len(names):
                raise ValueError(f"{name} names must be unique")
        if not self.action_fields:
            raise ValueError("action_fields must not be empty")
        if self.action_type == ActionType.EEF_POSE and (
            self.reference_frame is None or self.quaternion_order is None
        ):
            raise ValueError("eef_pose requires reference_frame and quaternion_order")
        return self


class ModelManifest(BoundaryProfile):
    model_id: Name
    source_kind: Literal["mock", "checkpoint"]
    training_robot_id: Name
    compatible_robot_ids: tuple[Name, ...] = ()
    predicted_horizon: Annotated[int, Field(gt=0, strict=True)]
    # Normalization belongs inside the policy; the public boundary is physical.
    state_space: Literal["raw_physical"] = "raw_physical"
    action_space: Literal["raw_physical"] = "raw_physical"
    normalization: Literal["none", "checkpoint"] = "none"
    checkpoint: ArtifactIdentity | None = None
    training_loader: ArtifactIdentity | None = None
    normalization_stats: ArtifactIdentity | None = None

    @model_validator(mode="after")
    def provenance(self):
        if self.source_kind == "checkpoint" and (
            self.checkpoint is None or self.training_loader is None
        ):
            raise ValueError("checkpoint manifests require checkpoint and training_loader identities")
        if self.normalization == "checkpoint" and self.normalization_stats is None:
            raise ValueError("checkpoint normalization requires a statistics identity")
        if self.source_kind == "mock" and self.checkpoint is not None:
            raise ValueError("a mock manifest cannot claim a checkpoint")
        return self


class RobotProfile(BoundaryProfile):
    profile_id: Name
    robot_id: Name
    control_hz: Rate
    # Explicit XPL grouping; no dimension-based or initial-pose reconstruction.
    legacy_state_groups: dict[str, tuple[Name, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def legacy_groups(self):
        if self.legacy_state_groups:
            flat = [name for names in self.legacy_state_groups.values() for name in names]
            state_names = [item.name for item in self.state_fields]
            if len(flat) != len(set(flat)) or set(flat) != set(state_names):
                raise ValueError("legacy_state_groups must cover every measured state field exactly once")
        return self


class Hdf5Column(Schema):
    key: Name
    column: Annotated[int, Field(ge=0, strict=True)]


class Hdf5Layout(Schema):
    state: dict[Name, Hdf5Column]
    action: dict[Name, Hdf5Column] = Field(default_factory=dict)
    timestamp_key: Name | None = None
    frame_id_key: Name | None = None
    instruction_key: Name = "instruction"
    instruction_mode: Literal["constant", "per_frame"] = "constant"
    instruction_index: Annotated[int, Field(ge=0, strict=True)] = 0


class DatasetProfile(BoundaryProfile):
    dataset_id: Name
    robot_id: Name
    artifact: ArtifactIdentity
    decoder_id: Name
    sample_hz: Rate
    clock_domain: Name
    source_kind: Literal["recorded", "synthetic"] = "recorded"
    hdf5: Hdf5Layout | None = None

    @model_validator(mode="after")
    def decoder_layout(self):
        if self.decoder_id == "xpl_hdf5_v1":
            if self.hdf5 is None:
                raise ValueError("xpl_hdf5_v1 requires an explicit hdf5 layout")
            if set(self.hdf5.state) != {field.name for field in self.state_fields}:
                raise ValueError("hdf5.state must map every named state field exactly")
            if self.hdf5.action and set(self.hdf5.action) != {field.name for field in self.action_fields}:
                raise ValueError("hdf5.action must map every named action field exactly")
            if any(camera.source_key is None for camera in self.cameras):
                raise ValueError("HDF5 cameras require explicit source_key paths")
            for columns in (self.hdf5.state, self.hdf5.action):
                bindings = [(value.key, value.column) for value in columns.values()]
                if len(bindings) != len(set(bindings)):
                    raise ValueError("HDF5 named fields cannot reuse a source column")
        return self


def _owned_array(value, *, ndim: int, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or any(dim == 0 for dim in array.shape):
        raise ValueError(f"{field_name} requires a non-empty {ndim}D array, got {array.shape}")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain finite real numeric values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Observation:
    images: Mapping[str, np.ndarray]
    state: np.ndarray
    state_names: tuple[str, ...]
    state_units: tuple[str, ...]
    instruction: str
    timestamp_s: float
    clock_domain: str
    frame_id: int
    episode_id: str

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self):
        if not isinstance(self.images, Mapping):
            raise TypeError("images must be a mapping of camera names to decoded RGB arrays")
        images = {}
        for name, value in self.images.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("camera names must be non-empty strings")
            image = np.asarray(value)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not image.size:
                raise ValueError(f"images.{name} must be a non-empty HWC uint8 RGB array")
            image = np.array(image, copy=True)
            image.setflags(write=False)
            images[name] = image
        object.__setattr__(self, "images", MappingProxyType(images))
        state = _owned_array(self.state, ndim=1, field_name="state")
        names, units = tuple(self.state_names), tuple(self.state_units)
        if any(not isinstance(x, str) or not x.strip() for x in names + units):
            raise ValueError("state_names and state_units require non-empty strings")
        if len(names) != state.size or len(units) != state.size or len(set(names)) != len(names):
            raise ValueError("state values, unique names and units must have the same length")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "state_units", units)
        if not isinstance(self.instruction, str):
            raise TypeError("instruction must be a string")
        if isinstance(self.timestamp_s, bool) or not isinstance(self.timestamp_s, Real) or not isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite and have an explicit clock_domain")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, Integral) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        for name in ("episode_id", "clock_domain"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_wire_dict(self) -> dict[str, Any]:
        # Existing PolicyServer owns decoding all vision/<camera>/color buffers.
        return {
            "schema_version": self.SCHEMA_VERSION,
            "vision": {name: {"color": value} for name, value in self.images.items()},
            "state": self.state, "state_names": list(self.state_names),
            "state_units": list(self.state_units), "instruction": self.instruction,
            "timestamp_s": self.timestamp_s, "clock_domain": self.clock_domain,
            "frame_id": self.frame_id, "episode_id": self.episode_id,
        }

    @classmethod
    def from_wire_dict(cls, data: Mapping[str, Any]) -> Observation:
        if not isinstance(data, Mapping) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
            raise ValueError("unsupported or missing Observation schema_version")
        allowed = {"schema_version", "vision", "state", "state_names", "state_units", "instruction", "timestamp_s", "clock_domain", "frame_id", "episode_id"}
        if set(data) != allowed:
            raise ValueError(f"Observation fields differ: missing={sorted(allowed-set(data))}, extra={sorted(set(data)-allowed)}")
        vision = data["vision"]
        if not isinstance(vision, Mapping):
            raise TypeError("vision must be a camera mapping")
        images = {}
        for name, value in vision.items():
            if not isinstance(value, Mapping) or set(value) != {"color"}:
                raise ValueError(f"vision.{name} must contain exactly the decoded color field")
            images[name] = value["color"]
        return cls(images=images, **{key: data[key] for key in allowed-{"schema_version", "vision"}})


class ContractValidator:
    """Compare named semantics; matching vector dimensions alone never suffices."""

    @staticmethod
    def profiles(model: ModelManifest, robot: RobotProfile, dataset: DatasetProfile | None = None) -> tuple[ContractIssue, ...]:
        issues = []

        def compare(path, expected, actual, code="semantic_mismatch"):
            if expected != actual:
                issues.append(ContractIssue(code, path, "declared semantics differ", expected, actual))

        if robot.robot_id not in (model.training_robot_id, *model.compatible_robot_ids):
            issues.append(ContractIssue("robot_incompatible", "robot.robot_id", "model has no compatibility declaration for this robot", model.training_robot_id, robot.robot_id))
        sources = [("robot", robot)] + ([("dataset", dataset)] if dataset is not None else [])
        for prefix, source in sources:
            cameras = {camera.name: camera for camera in source.cameras}
            for expected in model.cameras:
                actual = cameras.get(expected.name)
                if actual is None:
                    issues.append(ContractIssue("missing_camera", f"{prefix}.cameras.{expected.name}", "required camera is missing"))
                    continue
                compare(f"{prefix}.cameras.{expected.name}.color_space", expected.color_space, actual.color_space)
                if expected.shape is not None:
                    compare(f"{prefix}.cameras.{expected.name}.shape", expected.shape, actual.shape)
            if model.state_fields:
                compare(f"{prefix}.state_fields", [x.model_dump() for x in model.state_fields], [x.model_dump() for x in source.state_fields])
            compare(f"{prefix}.action_type", model.action_type.value, source.action_type.value)
            compare(f"{prefix}.action_fields", [x.model_dump() for x in model.action_fields], [x.model_dump() for x in source.action_fields])
            compare(f"{prefix}.reference_frame", model.reference_frame, source.reference_frame)
            compare(f"{prefix}.quaternion_order", model.quaternion_order, source.quaternion_order)
        if dataset is not None:
            compare("dataset.robot_id", model.training_robot_id, dataset.robot_id, "dataset_robot_mismatch")
        return tuple(issues)

    @classmethod
    def require_profiles(cls, model, robot, dataset=None):
        issues = cls.profiles(model, robot, dataset)
        if issues:
            raise ContractError(issues)

    @staticmethod
    def observation(obs: Observation, robot: RobotProfile, model: ModelManifest) -> None:
        issues = []
        for camera in model.cameras:
            image = obs.images.get(camera.name)
            if image is None:
                issues.append(ContractIssue("missing_camera", f"observation.images.{camera.name}", "required camera is missing"))
            elif camera.shape is not None and image.shape != camera.shape:
                issues.append(ContractIssue("image_shape", f"observation.images.{camera.name}", "boundary image shape differs", camera.shape, image.shape))
        names = tuple(x.name for x in robot.state_fields)
        units = tuple(x.unit for x in robot.state_fields)
        if obs.state_names != names or obs.state_units != units:
            issues.append(ContractIssue("state_layout", "observation.state", "named state order or units differ from RobotProfile", [names, units], [obs.state_names, obs.state_units]))
        if issues:
            raise ContractError(tuple(issues))
