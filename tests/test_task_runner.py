import json
from pathlib import Path
import time

import pytest
import yaml
from pydantic import ValidationError

from tools.create_mock_dataset import create_fixture
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.tasks import TaskInstruction, TaskSpec, SuccessEvaluator
from xpolicy.evaluation.lifecycle import StopSignal
from xpolicy.evaluation.runner import run_experiment
from xpolicy.policy.mock import MockPolicy


@pytest.fixture
def experiment(tmp_path):
    return create_fixture(tmp_path / "fixture", frames=6)


def configure(path, task=None, strategy=None):
    config = yaml.safe_load(path.read_text())
    if task is not None:
        config["task"] = task
    if strategy:
        config["execution"]["strategy"] = strategy
        config["execution"]["latency_trim"] = strategy != "sync"
    path.write_text(yaml.safe_dump(config))


def events(report):
    return [json.loads(line) for line in (Path(report["artifacts"]) / "trace.jsonl").read_text().splitlines()]


def test_task_override_is_explicit_and_native_reset_precedes_ready(experiment, monkeypatch):
    reset_ids, instructions = [], []
    reset, predict = MockPolicy.reset, MockPolicy.predict
    def track_reset(self, episode_id=None):
        reset_ids.append(episode_id)
        return reset(self, episode_id)
    def track_predict(self, obs):
        assert obs.episode_id in reset_ids
        instructions.append(obs.instruction)
        return predict(self, obs)
    monkeypatch.setattr(MockPolicy, "reset", track_reset)
    monkeypatch.setattr(MockPolicy, "predict", track_predict)
    configure(experiment, {"task_id": "declared_hold", "allowed_action_types": ["joint_position"],
                           "instruction": {"source": "override", "text": "explicit alternate instruction"},
                           "required_evidence": ["images", "commands", "predictions"]})
    report = run_experiment(experiment)
    assert report["status"] == "COMPLETED" and len(reset_ids) == 2
    assert set(instructions) == {"explicit alternate instruction"}
    for trial in report["trials"]:
        assert trial["task_id"] == "declared_hold"
        assert [item["state"] for item in trial["lifecycle"]["history"]] == [
            "CREATED", "PREFLIGHT", "PREPARED", "READY", "RUNNING", "STOPPING", "FINALIZING", "COMPLETED"]
        assert trial["metrics"]["inference_duration"]["samples"] == trial["prediction_requests"]
        assert trial["metrics"]["local_observation_age"]["samples"] == trial["prediction_requests"]
        assert trial["lifecycle"]["stop_reason"] == "dataset_end"
        assert trial["task_success"] is None
    observed = [item for item in events(report) if item["stage"] == "observation"]
    assert all(item["source_instruction"] == "hold the synthetic pose" for item in observed)
    assert all(item["instruction_source"] == "override" for item in observed)


@pytest.mark.parametrize("strategy", ["sync", "async_latest_only"])
def test_operator_stop_discards_returned_chunk_and_retains_unstarted_trial(experiment, monkeypatch, strategy):
    signal, calls = StopSignal(), []
    original = MockPolicy.predict
    def stop_during_predict(self, obs):
        calls.append(obs.episode_id)
        signal.request()
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", stop_during_predict)
    configure(experiment, strategy=strategy)
    report = run_experiment(experiment, stop_signal=signal)
    assert report["status"] == "STOPPED"
    assert len(calls) == len(report["trials"]) == len(report["not_started_trials"]) == 1
    trial = report["trials"][0]
    assert trial["recorded_action_steps"] == 0 and trial["fault"] is None
    assert trial["lifecycle"]["stop_reason"] == "operator_stop"
    assert report["not_started_trials"][0]["status"] == "NOT_RUN"
    assert trial["metrics"]["device_halt_latency"]["status"] == "NOT_MEASURED"
    assert trial["metrics"]["local_stop_latency"]["samples"] == 1


def test_preexisting_stop_does_not_construct_policy(experiment, monkeypatch):
    from xpolicy.evaluation import runner
    signal = StopSignal(); signal.request()
    def no_build(*args):
        raise AssertionError("policy must not be built after an existing stop")
    monkeypatch.setattr(runner, "build_policy", no_build)
    report = run_experiment(experiment, stop_signal=signal)
    assert report["status"] == "STOPPED" and report["trials"] == []
    assert len(report["not_started_trials"]) == 2


def test_task_timeout_cannot_extend_experiment_budget(experiment, monkeypatch):
    configure(experiment, {"task_id": "short", "allowed_action_types": ["joint_position"], "timeout_s": .02})
    original = MockPolicy.predict
    def slow(self, obs):
        time.sleep(.04)
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", slow)
    report = run_experiment(experiment)
    for trial in report["trials"]:
        assert trial["effective_timeout_s"] == .02
        assert trial["recorded_action_steps"] == 0
        assert trial["lifecycle"]["stop_reason"] == "timeout"
        assert trial["lifecycle"]["state"] == "FAILED"
        assert trial["metrics"]["inference_duration"]["samples"] == 1


def test_interrupt_writes_a_stopped_report(experiment, monkeypatch):
    def interrupt(self, obs):
        raise KeyboardInterrupt
    monkeypatch.setattr(MockPolicy, "predict", interrupt)
    report = run_experiment(experiment)
    assert report["status"] == "STOPPED"
    assert report["trials"][0]["lifecycle"]["stop_reason"] == "interrupted"
    assert (Path(report["artifacts"]) / "results.json").is_file()


@pytest.mark.parametrize("task,error", [
    ({"task_id": "bad_action", "allowed_action_types": ["joint_velocity"]}, ValueError),
    ({"task_id": "bad_score", "allowed_action_types": ["joint_position"], "evaluator": {"source": "ground_truth", "evaluator_id": "not_a_policy_outcome"}}, UnsupportedCapability),
    ({"task_id": "no_feedback", "allowed_action_types": ["joint_position"], "required_evidence": ["feedback"]}, UnsupportedCapability),
])
def test_task_admission_failure_writes_blocked_evidence(experiment, monkeypatch, task, error):
    from xpolicy.evaluation import runner
    monkeypatch.setattr(runner, "build_policy", lambda *args: pytest.fail("preflight must precede model construction"))
    configure(experiment, task)
    with pytest.raises(error) as failure:
        run_experiment(experiment)
    report = json.loads(Path(failure.value.report_path).read_text())
    assert report["status"] == "BLOCKED"
    assert all(item["lifecycle"]["state"] == "BLOCKED" for item in report["trials"])


def test_required_image_evidence_checks_configuration(experiment):
    configure(experiment, {"task_id": "images", "allowed_action_types": ["joint_position"], "required_evidence": ["images"]})
    config = yaml.safe_load(experiment.read_text()); config["artifacts"]["save_images"] = False
    experiment.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="decoded images"):
        run_experiment(experiment)


def test_required_commands_fail_if_async_replay_never_records_an_action(tmp_path, monkeypatch):
    experiment = create_fixture(tmp_path / "tiny", frames=1)
    configure(experiment, {"task_id": "needs_commands", "allowed_action_types": ["joint_position"], "required_evidence": ["commands"]}, "async_latest_only")
    original = MockPolicy.predict
    def slow(self, obs):
        time.sleep(.04)
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", slow)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert all(item["missing_required_evidence"] == ["commands"] for item in report["trials"])


def test_finished_trial_cannot_be_finalized_or_scored_twice(experiment):
    import io
    from xpolicy.core.config import load_experiment
    from xpolicy.evaluation.runner import _Trial
    config, contracts = load_experiment(experiment)
    trace = io.StringIO()
    trial = _Trial("idempotent", 0, config, contracts, None, experiment.parent, trace)
    trial.lifecycle.transition("PREFLIGHT")
    trial.block(ValueError("intentional block"))
    first = trial.finish()
    assert trial.finish() is first
    assert sum(json.loads(line)["stage"] == "trial_end" for line in trace.getvalue().splitlines()) == 1


def test_public_runner_evaluate_is_idempotent(experiment):
    from xpolicy import ExperimentRunner
    runner = ExperimentRunner(experiment)
    report = runner.evaluate()
    assert runner.evaluate() is report
    assert report["status"] == "COMPLETED"
    runner = ExperimentRunner(experiment)
    assert runner.stop() and not runner.stop()
    assert runner.evaluate()["status"] == "STOPPED"


def test_process_gate_stays_owned_until_pending_vendor_call_and_close_end(experiment, monkeypatch):
    import concurrent.futures
    import threading
    import numpy as np
    from xpolicy.evaluation import runner as runner_module
    configure(experiment, strategy="async_latest_only")
    config = yaml.safe_load(experiment.read_text())
    config["experiment"]["max_duration_s"] = .05
    experiment.write_text(yaml.safe_dump(config))
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = MockPolicy.predict
    def blocked(self, obs):
        entered.set()
        assert release.wait(10), "test must release the model worker"
        np.random.random()
        return original(self, obs)
    def close(self):
        np.random.random()
        closed.set()
    monkeypatch.setattr(MockPolicy, "predict", blocked)
    monkeypatch.setattr(MockPolicy, "close", close, raising=False)
    np.random.seed(234)
    previous = np.random.get_state()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_experiment, experiment)
        try:
            assert entered.wait(5)
            report = future.result(timeout=5)
            assert report["status"] == "FAULT" and report["policy_cleanup_pending"]
            assert not closed.is_set()
            with pytest.raises(UnsupportedCapability, match="still owns") as failure:
                run_experiment(experiment)
            refused = json.loads(Path(failure.value.report_path).read_text())
            assert refused["status"] == "BLOCKED"
            from xpolicy.evaluation.reports import summarize_reports
            assert summarize_reports([failure.value.report_path])["status"] == "BLOCKED"
        finally:
            release.set()
            assert closed.wait(5)
            # The close future callback restores RNG before releasing the lock.
            assert runner_module._EVALUATION_LOCK.acquire(timeout=5)
            runner_module._EVALUATION_LOCK.release()
    now = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(previous, now))


@pytest.mark.parametrize("strategy", ["sync", "async_latest_only"])
def test_policy_hooks_are_serial_and_final_hook_fault_is_retained(experiment, monkeypatch, strategy):
    configure(experiment, strategy=strategy)
    calls = []
    def prepare(self, meta):
        calls.append((meta["trial_id"], "prepare"))
    def end(self, result):
        assert result["task_success"] is None
        calls.append((result["trial_id"], "end"))
        raise RuntimeError("model final hook failed")
    monkeypatch.setattr(MockPolicy, "prepare_case", prepare, raising=False)
    monkeypatch.setattr(MockPolicy, "on_trial_end", end, raising=False)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT"
    assert [stage for _, stage in calls] == ["prepare", "end", "prepare", "end"]
    for trial in report["trials"]:
        assert trial["fault"]["message"] == "model final hook failed"
        assert trial["policy_hooks"] == {"prepare_case": "COMPLETED", "on_trial_end": "FAIL"}


def test_slow_async_prepare_hook_cannot_overlap_the_next_repeat(experiment, monkeypatch):
    configure(experiment, strategy="async_latest_only")
    config = yaml.safe_load(experiment.read_text())
    config["experiment"]["max_duration_s"] = .1
    experiment.write_text(yaml.safe_dump(config))
    calls = []
    def prepare(self, meta):
        calls.append(meta["trial_id"])
        time.sleep(.3)
    monkeypatch.setattr(MockPolicy, "prepare_case", prepare, raising=False)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT" and len(calls) == 1
    assert len(report["trials"]) == len(report["not_started_trials"]) == 1
    assert report["trials"][0]["policy_hooks"] == {"prepare_case": "FAIL", "on_trial_end": "NOT_RUN"}
    assert report["trials"][0]["inference_cleanup_drained"] is False


def test_cli_sigint_during_async_replay_retains_a_stopped_report(tmp_path):
    import signal
    import subprocess
    import sys
    experiment = create_fixture(tmp_path / "interrupt", frames=100, strategy="async_latest_only")
    process = subprocess.Popen([sys.executable, "-m", "xpolicy", "eval", "--config", str(experiment)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        ready = False
        while time.monotonic() < deadline and process.poll() is None:
            for trace in (experiment.parent / "results").glob("*/trace.jsonl"):
                ready = '"state": "RUNNING"' in trace.read_text()
            if ready:
                break
            time.sleep(.01)
        assert ready, "CLI did not reach RUNNING"
        process.send_signal(signal.SIGINT)
        out, error = process.communicate(timeout=10)
        assert process.returncode == 2, error
        report = json.loads(out)
        assert report["status"] == "STOPPED"
        assert report["trials"][0]["lifecycle"]["stop_reason"] == "interrupted"
        from xpolicy.evaluation.reports import summarize_reports
        assert summarize_reports([Path(report["artifacts"]) / "results.json"])["status"] == "STOPPED"
    finally:
        if process.poll() is None:
            process.kill(); process.communicate()


@pytest.mark.parametrize("kwargs", [{"source": "override"}, {"source": "dataset", "text": "ignored"}])
def test_instruction_contract_rejects_ambiguous_source(kwargs):
    with pytest.raises(ValidationError):
        TaskInstruction(**kwargs)


def test_evaluator_requires_identity_and_constraints_are_unique():
    with pytest.raises(ValidationError):
        SuccessEvaluator(source="human")
    with pytest.raises(ValidationError):
        TaskSpec(task_id="bad", allowed_action_types=["joint_position", "joint_position"])
