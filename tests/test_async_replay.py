import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from xpolicy.core.config import ExecutionConfig, ExperimentSettings, load_experiment
from xpolicy.evaluation.runner import run_experiment
from xpolicy.policy.mock import MockPolicy


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("async_fixture", ROOT / "tools/create_mock_dataset.py")
creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(creator)


@pytest.fixture
def experiment(tmp_path):
    path = creator.create_fixture(tmp_path / "fixture", frames=12)
    config = yaml.safe_load(path.read_text())
    config["execution"].update(strategy="async_latest_only", latency_trim=True, overlap_blend_steps=2)
    path.write_text(yaml.safe_dump(config))
    return path


def events(report):
    return [json.loads(line) for line in Path(report["artifacts"], "trace.jsonl").read_text().splitlines()]


def test_paced_async_replay_has_one_owner_and_reports_real_gaps(experiment):
    report = run_experiment(experiment)
    assert report["status"] == "COMPLETED", report
    assert report["execution_strategy"] == "async_latest_only"
    assert report["control_timebase"] == "paced_local_monotonic_control_ticks"
    assert [item["episode_generation"] for item in events(report) if item["stage"] == "execution_reset"] == [1, 2]
    for trial in report["trials"]:
        assert trial["causal"] is False
        assert trial["task_success"] is None
        assert trial["recorded_action_steps"] + trial["empty_buffer_ticks"] + trial["skipped_recorded_frames"] == trial["dataset_frames"]
        assert trial["inference_cleanup_drained"] is True
    trace = events(report)
    commands = [item for item in trace if item["stage"] == "command"]
    submitted = [item for item in trace if item["stage"] == "record_only_submission"]
    assert commands
    assert len(commands) == len(submitted)
    for command, submission in zip(commands, submitted):
        assert command["values"] == submission["values"][0]
        assert command["action_seq"] == submission["action_seq"]
        assert command["episode_generation"] == submission["episode_generation"]
        assert submission["causal"] is False
        assert submission["device_feedback"] is None
    for trial in report["trials"]:
        indices = [item["action_seq"] for item in commands if item["trial_id"] == trial["trial_id"]]
        assert indices == list(range(len(indices)))


def test_slow_model_supersedes_observations_and_drops_elapsed_predictions(experiment, monkeypatch):
    original = MockPolicy.predict
    def slow(self, obs):
        time.sleep(0.15)
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", slow)
    report = run_experiment(experiment)
    assert report["status"] == "COMPLETED", report
    assert all(item["recorded_action_steps"] == 0 for item in report["trials"])
    assert any(item["superseded_observations"] > 0 for item in report["trials"])
    trace = events(report)
    assert any(item.get("reason") == "prediction_horizon_elapsed" for item in trace)
    assert not any(item["stage"] == "command" for item in trace)


def test_async_inference_fault_and_cleanup_failure_are_retained(experiment, monkeypatch):
    def fail(self, obs):
        raise RuntimeError("async inference injected error")
    def close(self):
        raise RuntimeError("injected close error")
    monkeypatch.setattr(MockPolicy, "predict", fail)
    monkeypatch.setattr(MockPolicy, "close", close, raising=False)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert any(item["fault"]["message"] == "async inference injected error" for item in report["trials"])
    assert report["cleanup_errors"][0]["message"] == "injected close error"
    assert Path(report["artifacts"], "results.json").exists()


def test_sync_initialization_fault_is_written_before_model_exists(experiment, monkeypatch):
    from xpolicy.evaluation import runner
    config = yaml.safe_load(experiment.read_text())
    config["execution"]["strategy"] = "sync"
    experiment.write_text(yaml.safe_dump(config))
    def fail(contracts):
        raise RuntimeError("injected model startup error")
    monkeypatch.setattr(runner, "build_policy", fail)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert all(item["fault"]["message"] == "injected model startup error" for item in report["trials"])
    assert all(item["recorded_action_steps"] == 0 for item in report["trials"])
    assert Path(report["artifacts"], "results.json").exists()


def test_numpy_random_state_restored_after_policy_close(experiment, monkeypatch):
    def close(self):
        np.random.seed(1234)
    monkeypatch.setattr(MockPolicy, "close", close, raising=False)
    state = np.random.get_state()
    report = run_experiment(experiment)
    after = np.random.get_state()
    assert report["status"] == "COMPLETED"
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_model_call_after_budget_is_not_used_or_repeated(experiment, monkeypatch):
    config = yaml.safe_load(experiment.read_text())
    config["experiment"]["max_duration_s"] = 0.2
    experiment.write_text(yaml.safe_dump(config))
    calls, active, closed = [], [], []
    original = MockPolicy.predict
    def slow(self, obs):
        active.append(True)
        calls.append(obs.episode_id)
        time.sleep(0.5)
        try:
            return original(self, obs)
        finally:
            active.clear()
    def close(self):
        assert not active, "policy close raced with prediction"
        closed.append(True)
    monkeypatch.setattr(MockPolicy, "predict", slow)
    monkeypatch.setattr(MockPolicy, "close", close, raising=False)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert report["requested_repeats"] == 2
    assert len(report["trials"]) == 1
    assert report["trials"][0]["recorded_action_steps"] == 0
    assert report["trials"][0]["inference_cleanup_drained"] is False
    assert len(calls) == 1
    assert closed == [True]


def test_large_label_errors_are_scored_without_square_overflow(experiment):
    from xpolicy.evaluation.runner import _Trial
    from xpolicy.dataset.hdf5 import Hdf5Dataset
    config, contracts = load_experiment(experiment)
    source = Hdf5Dataset(config.environment.dataset_path, contracts.dataset)
    try:
        trial = _Trial("synthetic", 0, config, contracts, source, experiment.parent, io.StringIO())
        trial.errors = [np.full((2, 2), 1e308, dtype=np.longdouble)]
        result = trial.finish()
        assert result["status"] == "COMPLETED"
        for metric in result["behavior_cloning_error"].values():
            assert metric["mae"] == 1e308
            assert metric["rmse"] == 1e308
            assert metric["metric_status"] == "COMPUTED"
        trial.errors = [np.full((2, 2), np.longdouble("1e400"), dtype=np.longdouble)]
        result = trial.finish()
        assert result["status"] == "COMPLETED"
        for metric in result["behavior_cloning_error"].values():
            assert metric["mae"] is metric["rmse"] is None
            assert metric["metric_status"] == "EXCEEDS_FLOAT_RANGE"
    finally:
        source.close()


@pytest.mark.parametrize("changes", [{"latency_trim": 1}, {"latency_trim": "true"}, {"overlap_blend_steps": True}, {"overlap_blend_steps": 4}, {"empty_buffer": "hold"}])
def test_execution_configuration_is_explicit(changes):
    with pytest.raises(ValueError):
        ExecutionConfig(control_hz=30, execute_horizon=3, **changes)


@pytest.mark.parametrize("seed", [-1, 4294967296, True, "42"])
def test_invalid_seed_rejected_in_preflight(seed):
    with pytest.raises(ValueError):
        ExperimentSettings(name="bad_seed", seed=seed)
