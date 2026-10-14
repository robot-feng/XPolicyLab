"""Task intent and evidence constraints, independent of model/robot imports."""

from typing import Literal

from pydantic import Field, model_validator

from xpolicy.core.contracts import ActionType, ModelManifest, Name, Rate, Schema
from xpolicy.core.errors import UnsupportedCapability


class TaskInstruction(Schema):
    source: Literal["dataset", "override"] = "dataset"
    text: Name | None = None

    @model_validator(mode="after")
    def instruction_source(self):
        if (self.source == "override") != (self.text is not None):
            raise ValueError("instruction override requires text; dataset instructions must not supply text")
        return self


class SuccessEvaluator(Schema):
    source: Literal["not_measured", "ground_truth", "human", "heuristic"] = "not_measured"
    evaluator_id: Name | None = None

    @model_validator(mode="after")
    def evaluator_identity(self):
        if (self.source != "not_measured") != (self.evaluator_id is not None):
            raise ValueError("a measured evaluator requires an explicit evaluator_id")
        return self


class TaskSpec(Schema):
    task_id: Name
    instruction: TaskInstruction = Field(default_factory=TaskInstruction)
    timeout_s: Rate | None = None
    allowed_action_types: tuple[ActionType, ...] = Field(min_length=1)
    # This is a dataset cursor/model reset, never robot or scene homing.
    reset: Literal["recorded_episode_start"] = "recorded_episode_start"
    scene_id: Name | None = None
    evaluator: SuccessEvaluator = Field(default_factory=SuccessEvaluator)
    required_evidence: tuple[Literal["observations", "predictions", "commands", "images", "feedback"], ...] = ()

    @model_validator(mode="after")
    def unique_constraints(self):
        if len(set(self.allowed_action_types)) != len(self.allowed_action_types) or len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("task action types and evidence requirements must be unique")
        return self


def resolve_task(config, manifest: ModelManifest) -> TaskSpec:
    return config.task or TaskSpec(task_id=config.experiment.name, allowed_action_types=(manifest.action_type,))


def validate_task(config, manifest: ModelManifest) -> TaskSpec:
    task = resolve_task(config, manifest)
    if manifest.action_type not in task.allowed_action_types:
        raise ValueError("policy action_type is not allowed by TaskSpec")
    if config.environment.kind.startswith("dataset_"):
        if task.evaluator.source != "not_measured":
            raise UnsupportedCapability("recorded replay cannot judge the current policy's causal task success")
        if "feedback" in task.required_evidence:
            raise UnsupportedCapability("recorded replay provides no actuator feedback evidence")
    if set(task.required_evidence) & {"predictions", "commands"} and not config.artifacts.save_actions:
        raise ValueError("TaskSpec requires numeric action evidence but save_actions is false")
    if "images" in task.required_evidence and (not config.artifacts.save_images or config.validation.trace_level != "full"):
        raise ValueError("TaskSpec requires saved decoded images and trace_level=full")
    return task
