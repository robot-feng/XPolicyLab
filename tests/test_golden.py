import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from xpolicy.diagnostics.golden import compare_bundles, load_bundle, save_bundle


@pytest.fixture
def sample():
    return {
        "sample": {"dataset_id": "fixture", "episode_id": 3, "frame_id": 8,
                   "timestamp_s": 0.5, "clock_domain": "recorded"},
        "scope": "synthetic_fixture",
        "provenance": {"synthetic": True},
        "layers": [
            {"name": "O1", "metadata": {"camera_order": ["head"], "state_names": ["left", "right"], "state_units": ["rad", "rad"]},
             "arrays": {"image/head": np.full((2, 2, 3), [210, 40, 10], dtype=np.uint8), "state": np.array([[0.1, 0.2]], dtype=np.float32)},
             "axes": {"state": {"1": ["left", "right"]}}},
            {"name": "O2", "metadata": {"normalization": "synthetic_q99_once"},
             "arrays": {"state": np.array([[0.5, -0.5]], dtype=np.float16)},
             "axes": {"state": {"1": ["left", "right"]}}},
        ],
    }


def pair(tmp_path, sample, edit):
    expected = save_bundle(tmp_path / "expected", **sample)
    candidate = copy.deepcopy(sample)
    edit(candidate)
    observed = save_bundle(tmp_path / "observed", **candidate)
    return expected, observed


def test_verified_bundle_exact_comparison_and_ownership(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda value: value["provenance"].update(other=True))
    report = compare_bundles(expected, observed)
    assert report["status"] == "PASS"
    assert report["compared_layers"] == ["O1", "O2"]
    assert report["not_verified"] == ["physical_calibration", "checkpoint_forward", "device_behavior"]
    manifest, values = load_bundle(expected)
    state = values[manifest.layers[0].arrays["state"].key]
    assert not state.flags.writeable
    sample["layers"][0]["arrays"]["state"][:] = 999
    assert state[0, 0] == np.float32(0.1)


@pytest.mark.parametrize("key,value,field", [
    ("state_names", ["right", "left"], "metadata.state_names[0]"),
    ("state_units", ["deg", "rad"], "metadata.state_units[0]"),
    ("camera_order", [], "metadata.camera_order"),
])
def test_semantic_errors_precede_numeric_comparison(tmp_path, sample, key, value, field):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"][0]["metadata"].update({key: value}))
    report = compare_bundles(expected, observed)
    assert report["status"] == "FAIL"
    assert report["first_divergence"]["layer"] == "O1"
    assert report["first_divergence"]["field"] == field


def test_rgb_swap_is_localized_to_pixel_and_ignores_float_tolerance(tmp_path, sample):
    def edit(candidate):
        candidate["layers"][0]["arrays"]["image/head"] = candidate["layers"][0]["arrays"]["image/head"][..., ::-1]
    expected, observed = pair(tmp_path, sample, edit)
    difference = compare_bundles(expected, observed, atol=999)["first_divergence"]
    assert difference["layer"] == "O1"
    assert difference["field"] == "image/head"
    assert difference["index"] == [0, 0, 0]
    assert difference["expected"] == 210
    assert difference["observed"] == 10


def test_first_joint_error_and_earliest_layer(tmp_path, sample):
    def edit(candidate):
        candidate["layers"][0]["arrays"]["state"][0, 1] += 1
        candidate["layers"][1]["arrays"]["state"] *= 2
    expected, observed = pair(tmp_path, sample, edit)
    difference = compare_bundles(expected, observed)["first_divergence"]
    assert difference["layer"] == "O1"
    assert difference["labels"] == {"1": "right"}
    assert difference["index"] == [0, 1]


def test_double_normalization_is_o2_and_explicit_layer_scope(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"][1]["arrays"]["state"].__imul__(2))
    assert compare_bundles(expected, observed)["first_divergence"]["layer"] == "O2"
    report = compare_bundles(expected, observed, layers=["O1"])
    assert report["status"] == "PASS"
    assert "O2" in report["omitted_layers"]


def test_float_tolerance_and_dtype_are_explicit(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"][1]["arrays"]["state"].__iadd__(np.float16(0.001)))
    assert compare_bundles(expected, observed)["status"] == "FAIL"
    assert compare_bundles(expected, observed, atol=0.002)["status"] == "PASS"
    sample["layers"][1]["arrays"]["state"] = sample["layers"][1]["arrays"]["state"].astype(np.float32)
    third = save_bundle(tmp_path / "dtype", **sample)
    assert compare_bundles(expected, third)["first_divergence"]["kind"] == "dtype"


def test_missing_array_and_axis_labels(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"][0]["arrays"].pop("image/head"))
    assert compare_bundles(expected, observed)["first_divergence"]["kind"] == "array_names"
    sample["layers"][0]["axes"]["state"]["1"] = ["right", "left"]
    third = save_bundle(tmp_path / "axes", **sample)
    assert compare_bundles(expected, third)["first_divergence"]["kind"] == "axes"


def test_missing_o2_is_not_an_implicit_pass(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"].pop())
    assert compare_bundles(expected, observed)["status"] == "FAIL"
    assert compare_bundles(expected, observed, layers=["O1"])["status"] == "PASS"
    assert compare_bundles(expected, observed, layers=["O2"])["first_divergence"]["kind"] == "missing_layer"


def test_frame_clock_and_timestamp_types_must_match(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["sample"].update(frame_id=True))
    report = compare_bundles(expected, observed)
    assert report["first_divergence"]["layer"] == "sample_identity"
    assert report["first_divergence"]["kind"] == "metadata_type"


def test_hash_tampering_is_rejected(tmp_path, sample):
    path = save_bundle(tmp_path / "bundle", **sample)
    with (path / "arrays.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_bundle(path)


def test_per_array_content_hash_and_duplicate_json_rejected(tmp_path, sample):
    path = save_bundle(tmp_path / "bundle", **sample)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["layers"][0]["arrays"]["state"]["sha256"] = "0" * 64
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content differs"):
        load_bundle(path)
    (path / "manifest.json").write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate golden JSON"):
        load_bundle(path)


@pytest.mark.parametrize("value", [np.array([object()]), np.array([np.nan]), np.array([np.inf]), np.array([]), np.array(1)])
def test_unsafe_or_nonfinite_arrays_rejected(tmp_path, sample, value):
    sample["layers"][0]["arrays"]["state"] = value
    with pytest.raises(ValueError, match="nonempty finite"):
        save_bundle(tmp_path / "bundle", **sample)
    assert not (tmp_path / "bundle").exists()


def test_invalid_axes_order_tolerance_and_no_overwrite(tmp_path, sample):
    path = save_bundle(tmp_path / "bundle", **sample)
    with pytest.raises(FileExistsError):
        save_bundle(path, **sample)
    for value in (-1, float("nan"), True):
        with pytest.raises(ValueError, match="nonnegative"):
            compare_bundles(path, path, atol=value)
    with pytest.raises(ValueError, match="boundary order"):
        compare_bundles(path, path, layers=["O2", "O1"])
    sample["layers"][0]["axes"]["state"]["1"] = ["same", "same"]
    with pytest.raises(ValueError, match="axis labels"):
        save_bundle(tmp_path / "bad", **sample)


def test_cli_pass_and_difference_exit_codes(tmp_path, sample):
    expected, observed = pair(tmp_path, sample, lambda candidate: candidate["layers"][1]["arrays"]["state"].__imul__(2))
    command = [sys.executable, "-m", "xpolicy", "golden", "diff", "--expected", str(expected), "--observed", str(observed)]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 2, failed.stderr
    assert json.loads(failed.stdout)["first_divergence"]["layer"] == "O2"
    passed = subprocess.run(command + ["--layers", "O1"], capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(passed.stdout)["compared_layers"] == ["O1"]
