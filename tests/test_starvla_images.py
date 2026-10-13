import cv2
import numpy as np
import pytest
from PIL import Image

from XPolicyLab.policy.StarVLA.image_preprocessing import resize_rgb
from XPolicyLab.policy.StarVLA.model import Model


def image():
    return np.random.default_rng(42).integers(0, 256, (45, 63, 3), dtype=np.uint8)


def test_legacy_resize_is_unchanged_and_owned():
    rgb = image()
    rgb.setflags(write=False)
    expected = cv2.resize(rgb, (17, 13), interpolation=cv2.INTER_AREA)
    actual = resize_rgb(rgb, (17, 13))
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable


def test_training_pil_resize_matches_rgb_loader_default():
    rgb = image()
    rgb.setflags(write=False)
    expected = np.asarray(Image.fromarray(rgb).resize((17, 13)))
    actual = resize_rgb(rgb, (17, 13), "pillow_bicubic")
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable
    assert np.any(actual != resize_rgb(rgb, (17, 13), "opencv_area"))


def test_actual_adapter_honors_checkpoint_selected_resize():
    model = Model.__new__(Model)
    model.image_size, model.include_state = (17, 13), False
    model.model_cfg = {"image_resize_backend": "pillow_bicubic"}
    rgb = image()
    observation = {"vision": {name: {"color": rgb} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}, "instruction": "hold"}
    packed = model._convert_obs(observation)
    for value in packed["image"]:
        np.testing.assert_array_equal(value, np.asarray(Image.fromarray(rgb).resize((17, 13))))
    assert packed["lang"] == "hold"
    assert "state" not in packed


@pytest.mark.parametrize("size", [(0, 3), (3,), (True, 3), (3.5, 3)])
def test_invalid_image_size_rejected(size):
    with pytest.raises(ValueError, match="positive integers"):
        resize_rgb(image(), size)


def test_unknown_backend_and_encoded_buffers_rejected():
    with pytest.raises(ValueError, match="image_resize_backend"):
        resize_rgb(image(), (3, 3), "auto")
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        resize_rgb(np.array([1, 2, 3], dtype=np.uint8), (3, 3))
