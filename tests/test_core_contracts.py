from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from XPolicyLab.utils.contracts import ActionChunk as OriginalChunk
from xpolicy.core.contracts import ActionChunk, ContractValidator, ModelManifest, Observation, RobotProfile
from xpolicy.core.errors import ContractError
from xpolicy.policy.runtime import build_policy


def test_one_action_chunk_implementation():
    assert ActionChunk is OriginalChunk


def test_owned_observation_and_chunk(observation):
    image = observation.images["cam_head"].copy()
    state = observation.state.copy()
    owned = replace(observation, images={"cam_head": image}, state=state)
    image[:] = 0
    state[:] = 0
    np.testing.assert_array_equal(owned.images["cam_head"], observation.images["cam_head"])
    np.testing.assert_array_equal(owned.state, observation.state)
    with pytest.raises(ValueError):
        owned.state[0] = 0
    with pytest.raises(ValueError):
        owned.images["cam_head"][0, 0, 0] = 0
    values = np.ones((4, 2))
    chunk = ActionChunk(values, "joint_position")
    values[:] = 0
    assert chunk.values.sum() == 8
    with pytest.raises(ValueError):
        chunk.values[0, 0] = 0


@pytest.mark.parametrize("schema", [True, "1", 0, 2, None])
def test_wire_version_is_explicit(observation, schema):
    wire = observation.to_wire_dict()
    wire["schema_version"] = schema
    with pytest.raises(ValueError):
        Observation.from_wire_dict(wire)
    wire = ActionChunk(np.ones((4, 2)), "joint_position").to_wire_dict()
    wire["schema_version"] = schema
    with pytest.raises(ValueError):
        ActionChunk.from_wire_dict(wire)


@pytest.mark.parametrize("values", [np.zeros((0, 2)), np.zeros(2), [[True]], [[np.nan]], [[np.inf]], [[1j]], [["1"]]])
def test_invalid_action_values(values):
    with pytest.raises((TypeError, ValueError)):
        ActionChunk(values, "joint_position")


@pytest.mark.parametrize("change", [
    {"state": np.array([np.nan, 0.])}, {"frame_id": True}, {"frame_id": -1},
    {"timestamp_s": float("inf")}, {"timestamp_s": True}, {"clock_domain": ""},
    {"state_names": ("gripper", "gripper")}, {"state_units": ("rad",)},
    {"images": {"cam_head": np.zeros((3, 8, 8), dtype=np.uint8)}},
])
def test_invalid_observation(observation, change):
    with pytest.raises((TypeError, ValueError)):
        replace(observation, **change)


@pytest.mark.parametrize("change,code", [
    ({"images": {}}, "missing_camera"),
    ({"images": {"cam_head": np.zeros((9, 8, 3), dtype=np.uint8)}}, "image_shape"),
    ({"state_names": ("gripper", "arm_joint")}, "state_layout"),
    ({"state_units": ("deg", "m")}, "state_layout"),
])
def test_observation_semantic_errors(contracts, observation, change, code):
    with pytest.raises(ContractError) as error:
        ContractValidator.observation(replace(observation, **change), contracts.robot, contracts.manifest)
    assert code in {item.code for item in error.value.issues}


@pytest.mark.parametrize("change,path", [
    ({"robot_id": "different_robot"}, "robot.robot_id"),
    ({"cameras": []}, "robot.cameras.cam_head"),
    ({"action_type": "joint_delta"}, "robot.action_type"),
])
def test_profile_semantic_errors(contracts, change, path):
    robot = RobotProfile.model_validate({**contracts.robot.model_dump(), **change})
    issues = ContractValidator.profiles(contracts.manifest, robot)
    assert path in {issue.path for issue in issues}


def test_same_dimension_different_names_units_order_are_rejected(contracts):
    original = contracts.robot.model_dump()
    for fields in (
        list(reversed(original["action_fields"])),
        [{**original["action_fields"][0], "unit": "deg"}, original["action_fields"][1]],
        [{**original["action_fields"][0], "name": "right_arm"}, original["action_fields"][1]],
    ):
        robot = RobotProfile.model_validate({**original, "action_fields": fields})
        with pytest.raises(ContractError):
            ContractValidator.require_profiles(contracts.manifest, robot)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"predicted_horizon": True}, {"predicted_horizon": "4"},
    {"action_space": "normalized"}, {"normalization": "checkpoint"},
    {"source_kind": "checkpoint"}, {"cameras": [{"name": "cam_head", "color_space": "BGR"}]},
    {"action_fields": [{"name": "REQUIRED_EXPLICIT_MAPPING", "unit": "rad"}]},
])
def test_manifest_rejects_ambiguous_semantics(contracts, change):
    with pytest.raises(ValidationError):
        ModelManifest.model_validate({**contracts.manifest.model_dump(), **change})


def test_mock_full_chunk_one_prediction(contracts, observation):
    runtime = build_policy(contracts)
    chunk = runtime.predict_action_chunk(observation)
    assert runtime.model.calls == 1
    assert chunk.values.shape == (4, 2)
    np.testing.assert_array_equal(chunk.values, np.tile(observation.state, (4, 1)))
    assert (chunk.source_frame_id, chunk.episode_id, chunk.sequence_id) == (7, "episode-a", 0)
    assert chunk.execution_gaps == ("valid_until_s",)
    wire = chunk.to_wire_dict()
    wire["execution_ready"] = True
    assert not ActionChunk.from_wire_dict(wire).execution_ready
    assert runtime.predict_action_chunk(observation).sequence_id == 1
    assert runtime.model.calls == 2
    with pytest.raises(ValueError, match="RESET"):
        runtime.predict_action_chunk(replace(observation, episode_id="episode-b"))
    runtime.reset()
    chunk = runtime.predict_action_chunk(replace(observation, episode_id="episode-b"))
    assert chunk.sequence_id == 0
    assert runtime.model.calls == 1


def test_prediction_metadata_is_validated(contracts, observation):
    runtime = build_policy(contracts)
    predict = runtime.model.predict
    runtime.model.predict = lambda obs: replace(predict(obs), joint_names=("gripper", "arm_joint"))
    with pytest.raises(ContractError) as error:
        runtime.predict_action_chunk(observation)
    assert error.value.issues[0].path == "prediction.joint_names"
