"""Unified public API, sharing XPolicyLab's existing policy RPC and adapters."""

from xpolicy.core.contracts import (
    ActionChunk, ActionType, DatasetProfile, ModelManifest, Observation, RobotProfile,
)
from xpolicy.core.tasks import TaskSpec

__all__ = [
    "ActionChunk", "ActionType", "DatasetProfile", "ModelManifest", "Observation",
    "RobotProfile",
    "TaskSpec", "ExperimentRunner",
]


def __getattr__(name):
    if name == "ExperimentRunner":
        from xpolicy.evaluation.runner import ExperimentRunner
        return ExperimentRunner
    raise AttributeError(name)
