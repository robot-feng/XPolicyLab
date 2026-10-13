"""Deterministic CPU policy for contract and protocol evidence."""

import numpy as np

from xpolicy.core.contracts import ActionChunk, ActionType, ModelManifest, Observation


class MockPolicy:
    def __init__(self, manifest: ModelManifest):
        if manifest.source_kind != "mock":
            raise ValueError("MockPolicy requires a mock manifest")
        self.manifest = manifest
        self.calls = 0

    def reset(self, episode_id=None):
        self.calls = 0

    def predict(self, obs: Observation) -> ActionChunk:
        self.calls += 1
        indices = {name: index for index, name in enumerate(obs.state_names)}
        values = np.zeros(len(self.manifest.action_fields), dtype=np.float32)
        # This is a deterministic mock hold, not a trained task policy.
        if self.manifest.action_type == ActionType.JOINT_POSITION:
            for index, field in enumerate(self.manifest.action_fields):
                if field.name in indices and obs.state_units[indices[field.name]] == field.unit:
                    values[index] = obs.state[indices[field.name]]
        return ActionChunk(
            values=np.tile(values, (self.manifest.predicted_horizon, 1)),
            action_type=self.manifest.action_type,
            action_units=tuple(field.unit for field in self.manifest.action_fields),
            joint_names=tuple(field.name for field in self.manifest.action_fields) if self.manifest.action_type != ActionType.EEF_POSE else None,
            reference_frame=self.manifest.reference_frame,
            source_frame_id=obs.frame_id, episode_id=obs.episode_id,
            clock_domain=obs.clock_domain,
        )
