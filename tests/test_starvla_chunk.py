from types import SimpleNamespace

import numpy as np
import pytest

from XPolicyLab.policy.StarVLA.model import Model


def adapter(actions):
    # Exercise the real adapter with a counted model-server double. No
    # checkpoint or training-loader parity is claimed by this fixture.
    model = Model.__new__(Model)
    model.action_dim, model.action_chunk_size, model.execute_horizon = 2, 4, 2
    model.image_size, model.include_state = (8, 8), False
    model.use_ddim, model.num_ddim_steps, model.unnorm_key = True, 10, "mock_arm"
    model.model_cfg = {"action_units": ["rad", "m"], "joint_names": ["arm_joint", "gripper"]}
    model.obs_by_env, model.action_chunks_by_env, model.step_by_env = {}, {}, {}
    model._latest_env_idx_list = [0]
    calls = []
    def predict(payload):
        calls.append(payload)
        return {"ok": True, "data": {"actions": [actions]}}
    model.client = SimpleNamespace(predict_action=predict)
    return model, calls


def test_actual_adapter_chunk_does_not_consume_legacy_cursor(observation):
    model, calls = adapter(np.arange(8, dtype=np.float32).reshape(4, 2))
    obs = observation.to_wire_dict()
    obs["vision"]["cam_left_wrist"] = obs["vision"]["cam_head"]
    obs["vision"]["cam_right_wrist"] = obs["vision"]["cam_head"]
    obs["sequence_id"] = 3
    chunk = model.predict_action_chunk(obs)
    assert len(calls) == 1
    assert chunk.values.shape == (4, 2)
    assert chunk.source_frame_id == 7
    assert chunk.sequence_id == 3
    assert not model.obs_by_env and not model.action_chunks_by_env and not model.step_by_env
    model.update_obs(obs)
    np.testing.assert_array_equal(model._next_action_vector(0), chunk.values[0])
    np.testing.assert_array_equal(model._next_action_vector(0), chunk.values[1])
    assert len(calls) == 2
    model._next_action_vector(0)
    assert len(calls) == 3


@pytest.mark.parametrize("actions", [np.zeros((1, 2)), np.zeros((4, 3)), np.full((4, 2), np.nan)])
def test_actual_adapter_rejects_truncated_or_invalid_prediction(actions):
    model, calls = adapter(actions)
    with pytest.raises(ValueError):
        model._infer_converted_chunk({"lang": "hold", "image": []})
    assert len(calls) == 1
