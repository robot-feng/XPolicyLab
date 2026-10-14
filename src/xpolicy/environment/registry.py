"""Lazy environment discovery with explicit unavailable/deferred entries."""

from dataclasses import dataclass
from typing import Literal

from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.registry import Entry, Registry


@dataclass(frozen=True)
class EnvironmentEntry(Entry):
    status: Literal["IMPLEMENTED", "NOT_IMPLEMENTED", "DEFERRED"] = "IMPLEMENTED"
    reason: str | None = None
    causal_feedback: bool | None = None
    runtime_contract: str | None = None


class EnvironmentRegistry(Registry):
    def __init__(self):
        super().__init__("environment")

    def register(self, entry):
        if not isinstance(entry, EnvironmentEntry):
            raise ValueError("environment registration requires an EnvironmentEntry")
        if entry.status not in {"IMPLEMENTED", "NOT_IMPLEMENTED", "DEFERRED"}:
            raise ValueError("unknown environment implementation status")
        if entry.status == "IMPLEMENTED" and (not entry.factory or entry.reason is not None or not entry.runtime_contract):
            raise ValueError("implemented environment requires a factory/runtime contract and no blocking reason")
        if entry.status != "IMPLEMENTED" and (entry.factory or not entry.reason):
            raise ValueError("unavailable environments require a reason and must not expose a factory")
        super().register(entry)

    def require(self, name):
        entry = self.get(name)
        if entry.status != "IMPLEMENTED":
            raise UnsupportedCapability(f"environment provider unavailable: {name} ({entry.status}); {entry.reason}")
        return entry

    def create(self, name, *args, **kwargs):
        return self.require(name).load()(*args, **kwargs)


def environment_registry():
    registry = EnvironmentRegistry()
    registry.register(EnvironmentEntry(
        "dataset_direct", "xpolicy.environment.recorded:RecordedDatasetEnvironment",
        ("record_only", "sync", "async_latest_only", "finite_recorded_timeline", "label_error"),
        "XPolicyLab", causal_feedback=False, runtime_contract="recorded_timeline_v1"))
    for name, status, reason in (
        ("dataset_wire", "DEFERRED", "the genuine EVA/G2 WireObservation/ZMQ migration is deferred"),
        ("fake_plant", "DEFERRED", "the genuine EVA Fake G2 plant/strategy regression is deferred"),
        ("sdk_mock", "DEFERRED", "the migrated SDK Node injection and verified mappings are unavailable"),
        ("physics_sim", "NOT_IMPLEMENTED", "a genuine simulator provider, runtime/assets and original benchmark regression are required"),
        ("shadow", "DEFERRED", "device mappings and a verified NoActuationDriver have not been migrated"),
        ("real", "DEFERRED", "device-local guards, operator authorization and on-site acceptance are unavailable"),
    ):
        registry.register(EnvironmentEntry(name, "", origin="v4 blueprint", status=status, reason=reason))
    return registry
