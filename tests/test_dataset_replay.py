import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from xpolicy.core.config import load_experiment
from xpolicy.core.contracts import DatasetProfile
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.dataset.hdf5 import Hdf5Dataset
from xpolicy.evaluation.runner import _preflight, run_experiment
from xpolicy.execution.record_only import RecordOnlySink
from xpolicy.policy.runtime import build_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixture_creator", ROOT / "tools/create_mock_dataset.py")
fixture_creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_creator)


@pytest.fixture
def experiment(tmp_path):
    return fixture_creator.create_fixture(tmp_path / "fixture")


def test_replay_is_deterministic_noncausal_and_scores_labels(experiment):
    first = run_experiment(experiment)
    second = run_experiment(experiment)
    assert first["run_id"] != second["run_id"]
    assert first["status"] == second["status"] == "COMPLETED"
    assert first["dataset_source_kind"] == "synthetic"
    assert first["timebase_source"] == "stored"
    for trial in first["trials"] + second["trials"]:
        assert trial["recorded_action_steps"] == 5
        assert trial["predictions"] == 3
        assert trial["task_success"] is None
        assert trial["causal"] is False
        assert trial["behavior_cloning_error"]["arm_joint"]["mae"] > 0
        assert trial["behavior_cloning_error"]["arm_joint"]["unit"] == "rad"
    def events(report):
        return [json.loads(line) for line in (Path(report["artifacts"]) / "trace.jsonl").read_text().splitlines()]
    one, two = events(first), events(second)
    assert [item["state"] for item in one if item["stage"] == "observation"] == [item["state"] for item in two if item["stage"] == "observation"]
    for event in one:
        assert event["run_id"] == first["run_id"]
        assert event["trial_id"]
    predictions = [item for item in one if item["stage"] == "prediction"]
    submissions = [item for item in one if item["stage"] == "record_only_submission"]
    for prediction, submission in zip(predictions, submissions):
        assert prediction["chunk_id"] == submission["chunk_id"]
        assert prediction["values"][:submission["count"]] == submission["values"]


def test_dataset_row_seek_rgb_and_action_cannot_change_feedback(experiment):
    config, contracts = load_experiment(experiment)
    path = _preflight(config, contracts)
    source = Hdf5Dataset(path, contracts.dataset)
    try:
        first = source.observe(0, "episode")
        future = source.observe(2, "episode")
        assert first.images["cam_head"][0, 0, 0] > first.images["cam_head"][0, 0, 2]
        np.testing.assert_array_equal(first.state, source.observe(0, "episode").state)
        policy = build_policy(contracts)
        chunk = policy.predict_action_chunk(first)
        sink = RecordOnlySink()
        sink.submit(replace(chunk, values=np.full_like(chunk.values, 99)), 2)
        np.testing.assert_array_equal(future.state, source.observe(2, "episode").state)
    finally:
        source.close()


def test_missing_camera_or_timestamp_fault(experiment):
    config, contracts = load_experiment(experiment)
    with h5py.File(config.environment.dataset_path, "r+") as file:
        del file["vision/cam_head/colors"]
    with pytest.raises(ValueError, match="missing HDF5"):
        Hdf5Dataset(config.environment.dataset_path, contracts.dataset)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_experiment(experiment)


@pytest.mark.parametrize("kind", ["dataset_wire", "fake_plant", "sdk_mock", "physics_sim", "shadow", "real"])
def test_runner_rejects_unimplemented_modes_before_policy_import(experiment, kind):
    config, contracts = load_experiment(experiment)
    config = config.model_copy(update={"environment": config.environment.model_copy(update={"kind": kind})})
    with pytest.raises(UnsupportedCapability, match="environment provider unavailable"):
        _preflight(config, contracts)


def test_cli_replay(experiment):
    result = subprocess.run([sys.executable, "-m", "xpolicy", "eval", "--config", str(experiment)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "COMPLETED"
    assert Path(report["artifacts"], "results.json").exists()


def test_model_fault_is_recorded(experiment, monkeypatch):
    from xpolicy.policy.mock import MockPolicy
    def fail(self, obs):
        raise RuntimeError("injected inference failure")
    monkeypatch.setattr(MockPolicy, "predict", fail)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert all(item["recorded_action_steps"] == 0 for item in report["trials"])
    assert all(item["fault"]["message"] == "injected inference failure" for item in report["trials"])


@pytest.mark.parametrize("timestamps", [np.array([0., 0., 1., 2., 3.]), np.array([10, 9, 8, 7, 6], dtype=np.uint64), np.array([0., 1., np.nan, 3., 4.])])
def test_bad_recorded_timestamps_rejected(experiment, timestamps):
    config, contracts = load_experiment(experiment)
    with h5py.File(config.environment.dataset_path, "r+") as file:
        del file["timestamps"]
        file.create_dataset("timestamps", data=timestamps)
    with pytest.raises(ValueError, match="timestamps"):
        Hdf5Dataset(config.environment.dataset_path, contracts.dataset)


def test_duplicate_physical_source_binding_rejected(experiment):
    _, contracts = load_experiment(experiment)
    profile = contracts.dataset.model_dump()
    profile["hdf5"]["state"]["gripper"] = profile["hdf5"]["state"]["arm_joint"]
    with pytest.raises(ValueError, match="reuse a source column"):
        DatasetProfile.model_validate(profile)


def test_late_prediction_is_not_submitted(experiment, monkeypatch):
    from xpolicy.policy.mock import MockPolicy
    config = yaml.safe_load(experiment.read_text())
    config["experiment"]["max_duration_s"] = 0.02
    experiment.write_text(yaml.safe_dump(config))
    original = MockPolicy.predict
    def slow(self, obs):
        time.sleep(0.04)
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", slow)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert all(item["recorded_action_steps"] == 0 for item in report["trials"])
    assert all(item["fault"]["type"] == "TimeoutError" for item in report["trials"])
