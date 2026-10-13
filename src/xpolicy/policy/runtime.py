"""Explicit legacy or complete-chunk policy ownership; never mix both APIs."""

from __future__ import annotations

from dataclasses import replace

from xpolicy.core.artifacts import verify_artifact
from xpolicy.core.config import LoadedContracts
from xpolicy.core.contracts import ActionChunk, ContractValidator, Observation
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.registry import Registry, policy_registry


class LegacyPolicyShim:
    """Preserve a legacy adapter's result and its cursor without inventing a chunk."""

    def __init__(self, model):
        self.model = model

    def update_obs(self, observation):
        return self.model.update_obs(observation)

    def get_action(self):
        return self.model.get_action()

    def reset(self):
        return self.model.reset()

    def prepare_case(self, case_meta=None):
        hook = getattr(self.model, "prepare_case", None)
        return hook(case_meta) if callable(hook) else None

    def on_trial_end(self, result=None):
        hook = getattr(self.model, "on_trial_end", None)
        return hook(result) if callable(hook) else None

    def __getattr__(self, name):
        if name.startswith("_") or name in {"predict", "predict_action_chunk"}:
            raise AttributeError(name)
        return getattr(self.model, name)


def legacy_observation(obs: Observation, contracts: LoadedContracts, sequence_id: int) -> dict:
    groups = contracts.robot.legacy_state_groups
    if contracts.manifest.state_fields and not groups:
        raise UnsupportedCapability("a state-bearing legacy adapter requires explicit legacy_state_groups")
    indices = {name: index for index, name in enumerate(obs.state_names)}
    return {
        "vision": {name: {"color": image} for name, image in obs.images.items()},
        "state": {group: obs.state[[indices[name] for name in names]] for group, names in groups.items()},
        "instruction": obs.instruction, "frame_id": obs.frame_id,
        "episode_id": obs.episode_id, "sequence_id": sequence_id,
        "clock_domain": obs.clock_domain,
    }


class ChunkPolicyRuntime:
    def __init__(self, model, contracts: LoadedContracts, *, native: bool):
        ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
        self.model, self.contracts, self.native = model, contracts, native
        self._episode_id = None
        self._sequence_id = 0

    def describe(self):
        return {
            "api": "chunk", "manifest": self.contracts.manifest.model_dump(mode="json"),
            "robot_profile_id": self.contracts.robot.profile_id,
        }

    def reset(self):
        self._episode_id = None
        self._sequence_id = 0
        if not self.native:
            self.model.reset()

    def infer(self, observation):
        raise UnsupportedCapability("chunk services require CALL predict_action_chunk; legacy INFER is disabled")

    def predict_action_chunk(self, observation) -> ActionChunk:
        obs = observation if isinstance(observation, Observation) else Observation.from_wire_dict(observation)
        ContractValidator.observation(obs, self.contracts.robot, self.contracts.manifest)
        if self._episode_id is not None and self._episode_id != obs.episode_id:
            raise ValueError("episode_id changed without an explicit RESET")
        if self._episode_id is None:
            if self.native:
                self.model.reset(obs.episode_id)
            self._episode_id = obs.episode_id
        result = (
            self.model.predict(obs) if self.native else
            self.model.predict_action_chunk(legacy_observation(obs, self.contracts, self._sequence_id))
        )
        ContractValidator.prediction(result, self.contracts.manifest, source_frame_id=obs.frame_id,
                                     episode_id=obs.episode_id, sequence_id=self._sequence_id,
                                     clock_domain=obs.clock_domain, allow_missing_context=True)
        result = replace(result, source_frame_id=obs.frame_id, episode_id=obs.episode_id,
                         sequence_id=self._sequence_id, clock_domain=obs.clock_domain)
        self._sequence_id += 1
        return result

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


def build_policy(contracts: LoadedContracts, registry: Registry | None = None):
    ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
    registry = registry or policy_registry()
    entry = registry.get(contracts.policy.name)
    if contracts.policy.api not in entry.capabilities:
        raise UnsupportedCapability(f"policy {entry.name!r} does not declare {contracts.policy.api} support")
    if entry.name == "mock":
        if contracts.manifest.source_kind != "mock":
            raise ValueError("mock policy requires a mock manifest")
        model = entry.load()(contracts.manifest)
        return ChunkPolicyRuntime(model, contracts, native=True)
    if contracts.manifest.source_kind != "checkpoint":
        raise ValueError("legacy model adapters require a checkpoint manifest")
    # Verify provenance before importing vendor code or allocating model memory.
    checkpoint = verify_artifact(contracts.manifest.checkpoint, contracts.manifest_path)
    verify_artifact(contracts.manifest.training_loader, contracts.manifest_path)
    if contracts.manifest.normalization_stats is not None:
        verify_artifact(contracts.manifest.normalization_stats, contracts.manifest_path)
    if contracts.policy.checkpoint is not None and contracts.policy.checkpoint != checkpoint:
        raise ValueError("policy.checkpoint must match the verified manifest checkpoint")
    options = dict(contracts.policy.options)
    declared = {
        "policy_name": entry.name,
        "action_units": [field.unit for field in contracts.manifest.action_fields],
        "joint_names": [field.name for field in contracts.manifest.action_fields],
        "checkpoint_path": str(checkpoint),
    }
    for name, value in declared.items():
        if name in options and options[name] != value:
            raise ValueError(f"policy.options.{name} conflicts with the contract")
        options[name] = value
    model = entry.load()(options)
    if contracts.policy.api == "legacy":
        return LegacyPolicyShim(model)
    return ChunkPolicyRuntime(model, contracts, native=False)
