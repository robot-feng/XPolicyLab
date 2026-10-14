"""Unified public API, sharing XPolicyLab's existing policy RPC and adapters."""

from xpolicy.core.contracts import (
    ActionChunk, ActionType, DatasetProfile, ModelManifest, Observation, RobotProfile,
)
from xpolicy.core.tasks import TaskSpec

__all__ = [
    "ActionChunk", "ActionType", "DatasetProfile", "ModelManifest", "Observation",
    "RobotProfile",
    "TaskSpec", "ExperimentRunner", "ExperimentSuiteRunner",
]


def __getattr__(name):
    if name == "ExperimentRunner":
        from xpolicy.evaluation.runner import ExperimentRunner
        return ExperimentRunner
    if name == "ExperimentSuiteRunner":
        from xpolicy.evaluation.suite import ExperimentSuiteRunner
        return ExperimentSuiteRunner
    raise AttributeError(name)
