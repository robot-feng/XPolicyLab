import numpy as np
import pytest

from xpolicy.core.config import load_service
from xpolicy.integrations.eva_g2 import (
    decode_eva_observation,
    encode_fake_action,
    to_xpl_observation,
)


@pytest.fixture
def contracts():
    _, contracts = load_service("configs/integrations/eva_g2_fake/service.yaml")
    return contracts


@pytest.fixture
def eva_msgpack_numpy():
    return pytest.importorskip("openpi_client.msgpack_numpy")


def wire_payload(contracts, codec):
    robot = contracts.robot
    images = {
        camera.source_key: np.full(camera.shape, index + 1, dtype=np.uint8)
        for index, camera in enumerate(robot.cameras)
    }
    state = {}
    offset = 0
    for group, names in robot.legacy_state_groups.items():
        state[group] = np.arange(offset, offset + len(names), dtype=np.float32)
        offset += len(names)
    return codec.packb({"t": 12.5, "images": images, "state": state})


def test_eva_wire_frame_maps_named_groups_and_preserves_rgb_bytes(contracts, eva_msgpack_numpy):
    wire = decode_eva_observation(wire_payload(contracts, eva_msgpack_numpy))
    obs = to_xpl_observation(wire, contracts.robot, frame_id=9, episode_id="fake:0")

    assert obs.timestamp_s == 12.5
    assert obs.clock_domain == "eva_host_monotonic"
    assert obs.frame_id == 9 and obs.episode_id == "fake:0"
    assert tuple(obs.images) == ("front", "left_wrist", "right_wrist")
    np.testing.assert_array_equal(obs.images["front"], np.full((64, 64, 3), 1, dtype=np.uint8))
    np.testing.assert_array_equal(obs.images["left_wrist"], np.full((64, 64, 3), 2, dtype=np.uint8))
    assert obs.state_names == tuple(field.name for field in contracts.robot.state_fields)
    np.testing.assert_array_equal(obs.state, np.arange(24, dtype=np.float32))


@pytest.mark.parametrize("payload_kind", ["missing_timestamp", "nonfinite_timestamp", "invalid_images"])
def test_eva_wire_rejects_missing_or_invalid_boundary_fields(payload_kind, eva_msgpack_numpy):
    payloads = {
        "missing_timestamp": {"images": {}, "state": {}},
        "nonfinite_timestamp": {"t": float("nan"), "images": {}, "state": {}},
        "invalid_images": {"t": 1.0, "images": [], "state": {}},
    }
    with pytest.raises(ValueError):
        decode_eva_observation(eva_msgpack_numpy.packb(payloads[payload_kind]))


def test_eva_wire_rejects_camera_and_joint_group_mismatches(contracts, eva_msgpack_numpy):
    wire = decode_eva_observation(wire_payload(contracts, eva_msgpack_numpy))
    missing_camera = type(wire)(wire.timestamp_s, {}, wire.state)
    with pytest.raises(ValueError, match="missing configured camera"):
        to_xpl_observation(missing_camera, contracts.robot, frame_id=0, episode_id="fake:0")

    malformed_state = dict(wire.state)
    malformed_state["left_arm"] = np.zeros(7, dtype=np.float32)
    wrong_shape = type(wire)(wire.timestamp_s, wire.images, malformed_state)
    with pytest.raises(ValueError, match=r"expected \(8,\)"):
        to_xpl_observation(wrong_shape, contracts.robot, frame_id=0, episode_id="fake:0")


def test_fake_action_wire_is_finite_and_exactly_24d():
    codec = pytest.importorskip("openpi_client.msgpack_numpy")
    expected = np.linspace(-1, 1, 24, dtype=np.float32)
    decoded = codec.unpackb(encode_fake_action(expected))
    assert decoded["target"] == "real"
    assert decoded["action"].dtype == np.float32
    np.testing.assert_array_equal(decoded["action"], expected)

    with pytest.raises(ValueError, match="finite 24D"):
        encode_fake_action(np.zeros(23, dtype=np.float32))
    invalid = expected.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite 24D"):
        encode_fake_action(invalid)
