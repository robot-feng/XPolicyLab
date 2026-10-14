import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tools.create_mock_dataset import create_fixture
from xpolicy.evaluation.reports import summarize_reports
from xpolicy.evaluation.runner import run_experiment
from xpolicy.policy.mock import MockPolicy


@pytest.fixture
def experiment(tmp_path):
    return create_fixture(tmp_path / "fixture", frames=4)


def report_path(report):
    return Path(report["artifacts"]) / "results.json"


def test_summary_pools_raw_samples_and_keeps_unjudged_denominator(experiment):
    reports = [run_experiment(experiment) for _ in range(2)]
    summary = summarize_reports([report_path(item) for item in reports])
    assert summary["status"] == "PASS"
    assert summary["population"] == {"valid_runs": 2, "requested_trials": 4, "retained_trial_results": 4, "not_started_trials": 0}
    assert len(summary["groups"]) == 1
    group = summary["groups"][0]
    assert group["trials"] == group["execution_started_trials"] == 4
    assert group["task_score"]["success_rate"] is None
    assert group["task_score"]["unjudged_trials"] == 4
    metric = next(item for item in group["metrics"] if item["name"] == "inference_duration")
    values = [value for report in reports for trial in report["trials"] for value in trial["metrics"]["inference_duration"]["values"]]
    assert metric["samples"] == len(values) == 8
    assert metric["values"] == values
    assert group["execution_failure_rate"] == 0


def test_failed_run_is_not_hidden_by_a_healthy_run(experiment, monkeypatch):
    healthy = run_experiment(experiment)
    def fail(self, obs):
        raise ValueError("summary retains model failure")
    monkeypatch.setattr(MockPolicy, "predict", fail)
    failure = run_experiment(experiment)
    summary = summarize_reports([report_path(healthy), report_path(failure)])
    assert summary["status"] == "FAIL"
    assert [item["status"] for item in summary["inputs"]] == ["COMPLETED", "FAULT"]
    group = summary["groups"][0]
    assert group["outcome_counts"] == {"COMPLETED": 2, "FAULT": 2}
    assert group["execution_failure_rate"] == .5


def test_changed_task_conditions_are_not_averaged(experiment):
    first = run_experiment(experiment)
    config = yaml.safe_load(experiment.read_text())
    config["task"] = {"task_id": "different", "allowed_action_types": ["joint_position"]}
    experiment.write_text(yaml.safe_dump(config))
    second = run_experiment(experiment)
    summary = summarize_reports([report_path(first), report_path(second)])
    assert summary["status"] == "PASS" and len(summary["groups"]) == 2


def test_duplicate_report_does_not_count_trials_twice(experiment):
    report = report_path(run_experiment(experiment))
    summary = summarize_reports([report, report])
    assert summary["status"] == "FAIL"
    assert summary["population"]["valid_runs"] == 1
    assert "duplicate run_id" in summary["inputs"][1]["error"]["message"]


@pytest.mark.parametrize("filename", ["trace.jsonl", "config.json", "profiles.json"])
def test_mutated_evidence_is_rejected(experiment, filename):
    report = run_experiment(experiment)
    artifact = Path(report["artifacts"]) / filename
    artifact.write_text(artifact.read_text() + "\n")
    summary = summarize_reports([report_path(report)])
    assert summary["status"] == "FAIL" and summary["groups"] == []
    assert "SHA256 mismatch" in summary["inputs"][0]["error"]["message"]


@pytest.mark.parametrize("change", ["success", "mode", "metric", "path", "outcome"])
def test_results_metadata_cannot_disagree_with_hashed_evidence(experiment, change):
    path = report_path(run_experiment(experiment))
    report = json.loads(path.read_text())
    if change == "success":
        report["trials"][0]["task_success"] = True
        report["trials"][0]["success_source"] = "ground_truth"
    elif change == "mode":
        report["environment_kind"] = "real"
    elif change == "metric":
        report["trials"][0]["metrics"]["inference_duration"]["p95"] = 1234.
    elif change == "path":
        report["evidence_sha256"]["../outside"] = "0" * 64
    else:
        report["status"] = "FAULT"
    path.write_text(json.dumps(report))
    summary = summarize_reports([path])
    assert summary["status"] == "FAIL" and summary["groups"] == []


def test_blocked_mode_report_uses_common_format_and_retains_admission_failure(experiment):
    config = yaml.safe_load(experiment.read_text()); config["environment"]["kind"] = "fake_plant"
    experiment.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError) as error:
        run_experiment(experiment)
    summary = summarize_reports([error.value.report_path])
    assert summary["status"] == "BLOCKED"
    group = summary["groups"][0]
    assert group["execution_started_trials"] == 0
    assert group["admission_failures"] == 2
    assert group["execution_failure_rate"] is None
    assert group["conditions"]["environment_kind"] == "fake_plant"


def test_cli_summary_writes_common_report(experiment, tmp_path):
    report = report_path(run_experiment(experiment))
    output = tmp_path / "summary.json"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "results", "summarize", str(report), "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "PASS"


def test_legacy_or_duplicate_json_report_fails_without_masking_other_inputs(experiment, tmp_path):
    healthy = report_path(run_experiment(experiment))
    bad = tmp_path / "legacy.json"
    bad.write_text('{"schema_version":1,"schema_version":2}')
    summary = summarize_reports([healthy, bad])
    assert summary["status"] == "FAIL" and len(summary["groups"]) == 1
    assert "duplicate JSON key" in summary["inputs"][1]["error"]["message"]


@pytest.mark.parametrize("body", ['[]', '{"schema_version":2,"run_id":"bad","status":"COMPLETED"}', '{"schema_version":2,"run_id":"bad","status":NaN}'])
def test_invalid_json_structure_is_retained_as_a_failed_input(tmp_path, body):
    path = tmp_path / "broken.json"
    path.write_text(body)
    report = summarize_reports([path])
    assert report["status"] == "FAIL" and report["population"]["valid_runs"] == 0


def test_source_config_change_is_retained_as_evidence_failure(experiment, monkeypatch):
    original = MockPolicy.predict
    edited = False
    def edit_config(self, obs):
        nonlocal edited
        if not edited:
            experiment.write_text(experiment.read_text() + "\n# changed while running\n")
            edited = True
        return original(self, obs)
    monkeypatch.setattr(MockPolicy, "predict", edit_config)
    report = run_experiment(experiment)
    assert report["status"] == "FAULT" and not report["source_stable"]
    assert all(trial["status"] == "COMPLETED" for trial in report["trials"])
    summary = summarize_reports([report_path(report)])
    assert summary["status"] == "FAIL"
