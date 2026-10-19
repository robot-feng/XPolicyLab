"""ME-Dex Clean50 HDF5 images cross the shared RGB decoder boundary."""

import ast
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "policy/ME_Dex_1_0/training/data.py"


def _load_decode_frame():
    tree = ast.parse(LOADER.read_text(encoding="utf-8"))
    dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Clean50Dataset")
    method = next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_frame")
    isolated = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Clean50Dataset", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
    namespace = {"h5py": h5py, "np": np, "decode_image_bit": decode_image_bit}
    exec(compile(isolated, str(LOADER), "exec"), namespace)
    return namespace["Clean50Dataset"]._decode_frame


def _rgb_frame():
    frame = np.empty((24, 32, 3), dtype=np.uint8)
    frame[:] = [220, 34, 12]
    return frame


def _jpeg(frame, legacy):
    if legacy:
        # Synthetic historical fixture: RoboTwin passed its RGB array directly
        # to cv2.imencode, before the shared marker existed.
        success, payload = cv2.imencode(".jpg", frame)
        assert success
        return payload.tobytes()
    return encode_image_bit(frame)


@pytest.mark.parametrize("legacy", [False, True], ids=["marked-rgb", "legacy-robotwin"])
@pytest.mark.parametrize("layout", ["fixed-bytes", "vlen-uint8", "padded-matrix"])
def test_compressed_hdf5_layouts_decode_to_canonical_rgb(tmp_path, legacy, layout):
    frame = _rgb_frame()
    payload = _jpeg(frame, legacy)
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        if layout == "fixed-bytes":
            camera = handle.create_dataset("rgb", (1,), dtype=f"S{len(payload)}")
            camera[0] = payload
        elif layout == "vlen-uint8":
            camera = handle.create_dataset("rgb", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            camera[0] = np.frombuffer(payload, dtype=np.uint8)
        else:
            row = np.frombuffer(payload, dtype=np.uint8)
            camera = handle.create_dataset("rgb", (1, len(row) + 8), dtype=np.uint8)
            camera[0] = np.pad(row, (0, 8))
        actual = _load_decode_frame()(camera, 0)

    expected = decode_image_bit(payload)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.shape == (24, 32, 3)
    assert actual[..., 0].mean() > actual[..., 2].mean() + 100


def test_raw_rgb_array_is_returned_unchanged(tmp_path):
    frame = _rgb_frame()
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        camera = handle.create_dataset("rgb", data=frame[None])
        actual = _load_decode_frame()(camera, 0)
    np.testing.assert_array_equal(actual, frame)


def test_invalid_frame_reports_its_index(tmp_path):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        camera = handle.create_dataset("rgb", data=np.asarray([b"not-an-image"]))
        with pytest.raises(ValueError, match="camera frame at index 0"):
            _load_decode_frame()(camera, 0)


def test_training_loader_has_no_private_decoder_or_channel_swap():
    tree = ast.parse(LOADER.read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "decode_image_bit" in {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not attributes.intersection({"imdecode", "COLOR_BGR2RGB", "COLOR_RGB2BGR"})
