import numpy as np
import pytest

from XPolicyLab.utils.state_precision import cast_state, state_dtype, resolve_state_precision
from XPolicyLab.policy.StarVLA.runtime_config import validate_server_runtime_contract
from XPolicyLab.policy.StarVLA.model import Model


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_raw_precision_is_explicit_and_owned(dtype):
    raw = np.array([[0.0007884, 1.0]], dtype=np.float64)
    raw.setflags(write=False)
    value = cast_state(raw, dtype)
    assert value.dtype.name == dtype and value.flags.writeable
    assert not np.shares_memory(value, raw)


@pytest.mark.parametrize("dtype,normalized", [("float16", False), ("float64", True), ("auto", False), (np.float32, False), (True, True)])
def test_unknown_or_misplaced_precision_rejected(dtype, normalized):
    with pytest.raises(ValueError, match="dtype"):
        state_dtype(dtype, normalized=normalized)


@pytest.mark.parametrize("value", [[], [True], [complex(1)], [np.nan], [np.inf], np.zeros((1, 1, 1)), [1e100]])
def test_invalid_state_and_cast_overflow_rejected(value):
    with pytest.raises(ValueError, match="state"):
        cast_state(value, "float32")


def test_float16_model_quantization_is_retained():
    value = np.array([[0.123456789]], dtype=np.float64)
    packed = cast_state(value, "float16", normalized=True)
    assert packed.dtype == np.float16
    assert packed[0, 0] == np.float16(value[0, 0])
    with pytest.raises(ValueError, match="representable"):
        cast_state([[1e8]], "float16", normalized=True)


def metadata():
    return {"available_unnorm_keys": ["fixture"], "runtime_contract": {
        "version": 1, "image_color_order": "rgb", "state_input": "raw_env",
        "state_normalization": "training_transform", "action_output": "unnormalized_env",
        "action_dim": 2, "state_dim": 2,
    }}


def test_precision_handshake_preserves_old_float32_servers():
    args = dict(include_state=True, action_dim=2, unnorm_key="fixture")
    validate_server_runtime_contract(metadata(), **args)
    with pytest.raises(ValueError, match="state_input_dtype"):
        validate_server_runtime_contract(metadata(), state_input_dtype="float64", model_state_dtype="float16", **args)
    updated = metadata()
    updated["runtime_contract"].update(state_input_dtype="float64", model_state_dtype="float16")
    validate_server_runtime_contract(updated, state_input_dtype="float64", model_state_dtype="float16", **args)
    with pytest.raises(ValueError, match="model_state_dtype"):
        validate_server_runtime_contract(updated, state_input_dtype="float64", **args)


def test_actual_adapter_preserves_selected_raw_precision():
    # Synthetic single arm; no G2 head/waist field mapping is implied.
    model = Model.__new__(Model)
    model.model_cfg = {"state_input_dtype": "float64"}
    model.include_state, model.action_type, model.action_dim = True, "joint", 2
    model.robot_action_dim_info = {"arm_dim": [1], "ee_dim": [1]}
    model.image_size = (2, 2)
    raw = np.float64(0.123456789012345)
    obs = {"vision": {name: {"color": np.zeros((2, 2, 3), np.uint8)} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": {"joint_state": np.array([raw]), "ee_joint_state": np.array([0.])}}
    value = model._convert_obs(obs)["state"]
    assert value.dtype == np.float64 and value[0, 0] == raw
    model.model_cfg["state_input_dtype"] = "float32"
    assert model._convert_obs(obs)["state"].dtype == np.float32


def test_config_precision_precedence():
    config = {"state_input_dtype": "float64", "model_state_dtype": "float16"}
    assert resolve_state_precision({}, {}) == ("float32", "float32")
    assert resolve_state_precision(config, {}) == ("float64", "float16")
    assert resolve_state_precision(config, {"STARVLA_STATE_INPUT_DTYPE": "float32"}) == ("float32", "float16")
    assert resolve_state_precision(config, {"STARVLA_MODEL_STATE_DTYPE": ""}) == ("float64", "float16")


@pytest.mark.parametrize("config,env", [([], {}), ({}, []), ({"state_input_dtype": None}, {}), ({}, {"STARVLA_MODEL_STATE_DTYPE": "float64"})])
def test_invalid_precision_config_rejected(config, env):
    with pytest.raises(ValueError):
        resolve_state_precision(config, env)
