"""TinyVLA image-boundary checks, without loading its language backbone."""

import ast
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from XPolicyLab.policy.TinyVLA.image_preprocessing import read_camera_images
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit


ROOT = Path(__file__).resolve().parents[1]
LOADERS = [ROOT / "policy/TinyVLA/tinyvla/data_utils" / name for name in ("datasets.py", "utils.py")]


def rgb_frame():
    frame = np.empty((24, 32, 3), dtype=np.uint8)
    frame[:] = [218, 43, 12]
    frame[:, 16:] = [176, 65, 26]
    return frame


def encoded(frame, legacy):
    if legacy:
        # Deliberate historical test fixture: old producers passed RGB directly
        # to cv2.imencode. Published converted datasets use encode_image_bit.
        success, data = cv2.imencode(".jpg", frame)
        assert success
        return data.tobytes()
    return encode_image_bit(frame)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("layout", ["bytes", "uint8", "scalar", "singleton", "padded"])
@pytest.mark.parametrize("imsize", [32, 480])
def test_both_byte_formats_and_older_containers_decode_before_resize(legacy, layout, imsize):
    payload = encoded(rgb_frame(), legacy)
    value = payload
    if layout == "uint8":
        value = np.asarray(bytearray(payload), dtype=np.uint8)
    elif layout == "scalar":
        value = np.array(payload, dtype=object)
    elif layout == "singleton":
        value = np.array([payload], dtype=object)
    elif layout == "padded":
        value = np.asarray(bytearray(payload + b"\0" * 12), dtype=np.uint8)
    result = read_camera_images({"cam": [value]}, ("cam",), 0, imsize)["cam"]
    expected = decode_image_bit(payload)
    if imsize != expected.shape[1]:
        expected = cv2.resize(expected, (320, 180), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result, expected)
    assert result[..., 0].mean() > result[..., 2].mean() + 100
    assert result.dtype == np.uint8 and result.flags.c_contiguous


@pytest.mark.parametrize("imsize", [32, 480])
def test_raw_rgb_resize_matches_existing_uncompressed_loader(imsize):
    frame = rgb_frame()
    frame.setflags(write=False)
    expected = frame if imsize == frame.shape[1] else cv2.resize(frame, (320, 180))
    result = read_camera_images({"cam": [frame]}, ("cam",), 0, imsize)["cam"]
    np.testing.assert_array_equal(result, expected)
    assert not frame.flags.writeable


@pytest.mark.parametrize("value", [b"invalid image", np.zeros((24, 32, 4), np.uint8), np.zeros((24, 32, 3), np.float32),
                                  np.zeros((2, 24, 32, 3), np.uint8)])
def test_invalid_or_multiple_frames_are_rejected_with_camera_context(value):
    with pytest.raises(ValueError, match="TinyVLA camera 'wrist', frame 0"):
        read_camera_images({"wrist": [value]}, ("wrist",), 0, 32)


@pytest.mark.parametrize("imsize", [True, 0, None, 3.5])
def test_invalid_width_is_rejected(imsize):
    with pytest.raises(ValueError, match="positive integer"):
        read_camera_images({"cam": [rgb_frame()]}, ("cam",), 0, imsize)


@pytest.mark.parametrize("path", LOADERS, ids=lambda path: path.name)
def test_both_loader_definitions_share_the_rgb_boundary(path):
    tree = ast.parse(path.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "XPolicyLab.policy.TinyVLA.image_preprocessing" and any(alias.name == "read_camera_images" for alias in node.names) for node in imports)
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"imdecode", "COLOR_BGR2RGB", "COLOR_RGB2BGR"} for node in ast.walk(tree))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EpisodicDataset")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "read_camera_images" for node in ast.walk(method)) == 1


@pytest.mark.parametrize("path", LOADERS, ids=lambda path: path.name)
@pytest.mark.parametrize("storage", ["raw", "marked", "legacy"])
@pytest.mark.parametrize("compress_flag", [False, True])
def test_actual_loader_getitem_image_tensor_with_real_torch(path, storage, compress_flag, tmp_path):
    torch = pytest.importorskip("torch")
    # Execute the actual source methods with real Torch/HDF5. This isolates the
    # changed data path from LLaVA imports and bypasses constructor/augmentation;
    # it is not a full TinyVLA training or checkpoint-constructor acceptance.
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EpisodicDataset")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"__getitem__", "_locate_transition"}]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "np": np, "h5py": h5py, "read_camera_images": read_camera_images}
    exec(compile(module, str(path), "exec"), namespace)
    dataset = namespace["EpisodicDataset"]()
    fixture = tmp_path / "episode.hdf5"
    names = ("left", "right", "head")
    expected_frames = []
    with h5py.File(fixture, "w") as file:
        file.attrs["sim"], file.attrs["compress"] = True, compress_flag
        file.create_dataset("language_raw", data=np.asarray([b"hold synthetic pose"]))
        for key in ("action", "observations/qpos", "observations/qvel"):
            file.create_dataset(key, data=np.zeros((2, 2), np.float32))
        for index, name in enumerate(names):
            frame = rgb_frame().copy()
            frame[..., 1] += index * 10
            if storage == "raw":
                file.create_dataset(f"observations/images/{name}", data=np.stack([frame, frame]))
                expected_frames.append(frame)
            else:
                payload = encoded(frame, storage == "legacy")
                field = file.create_dataset(f"observations/images/{name}", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                field[0] = field[1] = np.asarray(bytearray(payload), dtype=np.uint8)
                expected_frames.append(decode_image_bit(payload))
    dataset.dataset_path_list, dataset.episode_ids = [str(fixture)], [0]
    dataset.episode_len, dataset.cumulative_len = np.array([2]), np.array([2])
    dataset.max_episode_len, dataset.chunk_size = 2, 2
    dataset.camera_names, dataset.imsize = names, 32
    dataset.transformations, dataset.augment_images, dataset.policy_class = [], False, "ACT"
    dataset.norm_stats = {"qpos_mean": torch.zeros(2), "qpos_std": torch.ones(2), "action_mean": torch.zeros(2), "action_std": torch.ones(2)}
    images, state, actions, padding = dataset[0]
    expected = torch.from_numpy(np.stack(expected_frames)).permute(0, 3, 1, 2) / 255.
    torch.testing.assert_close(images, expected, atol=0, rtol=0)
    assert images.shape == (3, 3, 24, 32) and state.shape == (2,) and actions.shape == (2, 2)
    assert not padding.any()


def test_before_after_diagnostic_retains_real_baseline_counterexamples(tmp_path):
    pytest.importorskip("torch")
    from tools.validate_tinyvla_image_boundary import validate
    report = validate(tmp_path / "before-after")
    assert report["status"] == "PASS" and report["source_stable"]
    assert report["fixture_source_kind"] == "synthetic"
    assert len(report["cases"]) == 10 and report["baseline_failures_retained"] == 8
    assert all(report["counterexamples_reproduced"].values())
    assert all(case["current"]["status"] == "PASS" for case in report["cases"])
    assert "checkpoint_inference" in report["not_run"]


def test_missing_git_baseline_is_blocked_in_retained_diagnostic(tmp_path):
    pytest.importorskip("torch")
    from tools.validate_tinyvla_image_boundary import validate
    report = validate(tmp_path / "missing-baseline", "not-a-materialized-baseline-ref")
    assert report["status"] == "BLOCKED" and report["source_stable"]
    assert report["cases"] == []
