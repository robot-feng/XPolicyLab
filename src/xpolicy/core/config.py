"""One set of strict configuration schemas, resolved before model imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, StrictBool, field_validator, model_validator

from xpolicy.core.contracts import DatasetProfile, ModelManifest, Name, Rate, RobotProfile, Schema
from xpolicy.core.tasks import TaskSpec


class ServerSettings(Schema):
    host: Name = "127.0.0.1"
    port: int = Field(default=19000, ge=0, le=65535, strict=True)
    ws_ping_interval_s: float | None = Field(default=20.0, ge=0, allow_inf_nan=False, strict=True)
    ws_ping_timeout_s: float | None = Field(default=20.0, ge=0, allow_inf_nan=False, strict=True)


class PolicyConfig(Schema):
    name: Name
    api: Literal["chunk", "legacy"] = "chunk"
    manifest: Path
    checkpoint: Path | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PolicyServiceConfig(Schema):
    policy: PolicyConfig
    robot: Path
    server: ServerSettings = Field(default_factory=ServerSettings)


class ExperimentSettings(Schema):
    name: Name
    seed: int = Field(default=0, ge=0, le=4294967295, strict=True)
    repeats: int = Field(default=1, ge=1, strict=True)
    max_duration_s: Rate = 60.0


class EnvironmentConfig(Schema):
    kind: Literal["dataset_direct", "dataset_wire", "fake_plant", "sdk_mock", "physics_sim", "shadow", "real"]
    robot: Path
    dataset_profile: Path | None = None
    dataset_path: Path | None = None
    episode: int = Field(default=0, ge=0, strict=True)


class ExecutionConfig(Schema):
    mode: Literal["record_only", "shadow", "real"] = "record_only"
    strategy: Literal["sync", "async_latest_only"] = "sync"
    control_hz: Rate
    execute_horizon: int = Field(ge=1, strict=True)
    latency_trim: StrictBool = False
    overlap_blend_steps: int = Field(default=0, ge=0, strict=True)
    # This is the only implemented empty-buffer action in a record-only sink.
    empty_buffer: Literal["skip_record"] = "skip_record"

    @model_validator(mode="after")
    def overlap_window(self):
        if self.overlap_blend_steps > self.execute_horizon:
            raise ValueError("overlap_blend_steps cannot exceed execute_horizon")
        return self


class ValidationConfig(Schema):
    strict_contract: Literal[True] = True
    reject_missing_cameras: Literal[True] = True
    reject_unmapped_joints: Literal[True] = True
    trace_level: Literal["full", "summary"] = "full"

    @field_validator("strict_contract", "reject_missing_cameras", "reject_unmapped_joints", mode="before")
    @classmethod
    def require_true(cls, value):
        if value is not True:
            raise ValueError("contract checks must be enabled with a YAML boolean true")
        return value


class ArtifactsConfig(Schema):
    root: Path = Path("runs")
    save_images: StrictBool = True
    save_actions: StrictBool = True
    save_feedback: StrictBool = True


class ExperimentConfig(Schema):
    experiment: ExperimentSettings
    policy: PolicyConfig
    environment: EnvironmentConfig
    execution: ExecutionConfig
    task: TaskSpec | None = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    @model_validator(mode="after")
    def mode_agreement(self):
        expected = {"shadow": "shadow", "real": "real"}.get(self.environment.kind, "record_only")
        if self.execution.mode != expected:
            raise ValueError(f"{self.environment.kind} requires execution.mode={expected}")
        if self.environment.kind.startswith("dataset_") and (
            self.environment.dataset_profile is None or self.environment.dataset_path is None
        ):
            raise ValueError("dataset modes require dataset_profile and dataset_path")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        try:
            value = yaml.load(stream, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return value


def resolve_path(path: Path, parent: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


@dataclass(frozen=True)
class LoadedContracts:
    policy: PolicyConfig
    manifest: ModelManifest
    robot: RobotProfile
    dataset: DatasetProfile | None
    source_path: Path
    manifest_path: Path
    robot_path: Path
    dataset_profile_path: Path | None = None


def _load_contracts(policy: PolicyConfig, robot_path: Path, source: Path, dataset_path=None) -> LoadedContracts:
    manifest_path = resolve_path(policy.manifest, source.parent)
    robot_path = resolve_path(robot_path, source.parent)
    checkpoint = resolve_path(policy.checkpoint, source.parent) if policy.checkpoint is not None else None
    policy = policy.model_copy(update={"manifest": manifest_path, "checkpoint": checkpoint})
    dataset_path = resolve_path(dataset_path, source.parent) if dataset_path is not None else None
    return LoadedContracts(
        policy, ModelManifest.model_validate(read_yaml(manifest_path)),
        RobotProfile.model_validate(read_yaml(robot_path)),
        DatasetProfile.model_validate(read_yaml(dataset_path)) if dataset_path is not None else None,
        source, manifest_path, robot_path, dataset_path,
    )


def load_service(path: str | Path) -> tuple[PolicyServiceConfig, LoadedContracts]:
    source = Path(path).expanduser().resolve()
    config = PolicyServiceConfig.model_validate(read_yaml(source))
    contracts = _load_contracts(config.policy, config.robot, source)
    return config.model_copy(update={"policy": contracts.policy, "robot": contracts.robot_path}), contracts


def load_experiment(path: str | Path) -> tuple[ExperimentConfig, LoadedContracts]:
    source = Path(path).expanduser().resolve()
    config = ExperimentConfig.model_validate(read_yaml(source))
    contracts = _load_contracts(config.policy, config.environment.robot, source, config.environment.dataset_profile)
    environment = config.environment.model_copy(update={
        "robot": contracts.robot_path,
        "dataset_profile": contracts.dataset_profile_path,
        "dataset_path": resolve_path(config.environment.dataset_path, source.parent) if config.environment.dataset_path is not None else None,
    })
    config = config.model_copy(update={
        "policy": contracts.policy, "environment": environment,
        "artifacts": config.artifacts.model_copy(update={"root": resolve_path(config.artifacts.root, source.parent)}),
    })
    return config, contracts
