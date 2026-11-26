"""Joint-action candidates from an observed demonstration and a frozen policy plan."""

import numpy as np

LEADS = (0, 2, 4, 8)


def candidates(demo, current, base, phase, executed):
    # Only search forward near the last confirmed phase, including on paths that cross themselves.
    stop = min(len(demo), phase + 2 * executed + 1)
    phase += int(np.argmin(np.square(demo[phase:stop] - current).mean(axis=1)))
    indices = phase + np.arange(1, len(base) + 1)[:, None] + np.asarray(LEADS)
    reference = demo[np.minimum(indices, len(demo) - 1)].transpose(0, 2, 1)
    return np.concatenate((base[..., None], reference), axis=-1), phase


class DemoHead:
    """A convex joint controller fitted on pairs of observed demonstrations and training actions."""

    def __init__(self, path):
        import json
        from pathlib import Path

        artifact = json.loads(Path(path).read_text())
        self.weights = np.asarray(artifact["weights"], dtype=np.float64)
        if (
            artifact["version"] != 1
            or tuple(artifact["leads"]) != LEADS
            or self.weights.shape != (len(LEADS) + 1,)
            or not np.isfinite(self.weights).all()
            or np.any(self.weights < 0)
            or not np.isclose(self.weights.sum(), 1)
        ):
            raise ValueError("Invalid joint demonstration head")

    def apply(self, demo, current, action, state, processor, phase, executed):
        import torch
        from dataclasses import replace
        from XPolicyLab.policy.G05.model import _to_horizon_array

        base = _to_horizon_array(action["right_arm"])
        choices, phase = candidates(demo, current, base, phase, executed)
        arm = (choices @ self.weights).astype(np.float32)
        corrected = {**action, "right_arm": arm}
        # The next receipt must commit the dispatched commands, including their gripper values.
        parts = {
            meta["key"]: torch.from_numpy(
                _to_horizon_array(corrected[meta["key"]]).copy()
            )
            for meta in processor.shape_meta["action"]
        }
        batch = processor.normalizer.forward({"action": parts})
        normalized = processor.action_state_merger.forward(batch)["action"]
        pending = state.pending_action
        entry = pending["action"]
        if normalized.shape != entry["value"].shape:
            raise ValueError(
                "Demonstration head and pending plan have different action layouts"
            )
        state = replace(
            state,
            pending_action={
                **pending,
                "action": {**entry, "value": normalized.to(entry["value"])},
            },
        )
        return corrected, state, phase
