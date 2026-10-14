import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest
import yaml
from pydantic import ValidationError

from tools.create_mock_dataset import create_fixture
from xpolicy import ExperimentSuiteRunner
from xpolicy.core.process import run_owned_process
from xpolicy.evaluation.suite import SuiteConfig, verify_suite


@pytest.fixture
def suite(tmp_path):
    experiment = create_fixture(tmp_path / "fixture", frames=4)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "name": "synthetic_suite", "artifacts_root": "suite-results",
                                    "process_timeout_s": 20., "cleanup_grace_s": .1,
                                    "cases": [{"case_id": "baseline", "config": str(experiment.relative_to(tmp_path)),
                                               "seeds": [3, 7], "repeats": 1}]}))
    return path


def configure(path, **changes):
    config = yaml.safe_load(path.read_text())
    config.update(changes)
    path.write_text(yaml.safe_dump(config))


def test_actual_suite_runs_seeds_in_separate_runner_processes_and_verifies_evidence(suite):
    before = (suite.parent / "fixture/experiment.yaml").read_bytes()
    runner = ExperimentSuiteRunner(suite)
    report = runner.evaluate()
    assert runner.evaluate() is report
    assert report["status"] == "PASS" and report["source_stable"]
    assert report["population"] == {"planned_cases": 2, "processes_started": 2, "outcome_counts": {"COMPLETED": 2}, "verified_runs": 2}
    assert len({case["process"]["pid"] for case in report["cases"]}) == 2
    assert [case["seed"] for case in report["cases"]] == [3, 7]
    assert report["summary"]["population"]["requested_trials"] == 2
    assert report["summary"]["groups"][0]["seeds"] == [3, 7]
    assert report["summary"]["groups"][0]["task_score"]["success_rate"] is None
    assert (suite.parent / "fixture/experiment.yaml").read_bytes() == before
    path = suite.parent / "suite-results/SUITE_REPORT.json"
    assert verify_suite(path) == report
    moved = suite.parent / "moved-suite"
    shutil.move(path.parent, moved)
    assert verify_suite(moved / "SUITE_REPORT.json") == report


def test_multiple_case_conditions_are_kept_in_separate_summary_groups(suite):
    original = suite.parent / "fixture/experiment.yaml"
    alternate = suite.parent / "fixture/alternate.yaml"
    config = yaml.safe_load(original.read_text())
    config["policy"]["options"] = {"step_delta": .02}
    alternate.write_text(yaml.safe_dump(config))
    configure(suite, cases=[{"case_id": "first", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1},
                            {"case_id": "second", "config": "fixture/alternate.yaml", "seeds": [3], "repeats": 1}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "PASS"
    assert len(report["summary"]["groups"]) == 2
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


@pytest.mark.parametrize("fail_fast", [False, True])
def test_blocked_case_keeps_result_and_fail_fast_preserves_unstarted_cases(suite, fail_fast):
    original = suite.parent / "fixture/experiment.yaml"
    config = yaml.safe_load(original.read_text())
    config["environment"]["kind"] = "physics_sim"
    blocked = original.with_name("blocked.yaml")
    blocked.write_text(yaml.safe_dump(config))
    configure(suite, fail_fast=fail_fast,
              cases=[{"case_id": "blocked", "config": "fixture/blocked.yaml", "seeds": [3], "repeats": 1},
                     {"case_id": "healthy", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "BLOCKED"
    assert report["cases"][0]["status"] == "BLOCKED"
    assert report["cases"][0]["process"]["exit_code"] == 2
    assert report["cases"][1]["status"] == ("NOT_RUN" if fail_fast else "COMPLETED")
    assert report["population"]["planned_cases"] == 2
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_bad_case_configuration_does_not_disappear_or_prevent_later_cases(suite):
    configure(suite, cases=[{"case_id": "missing", "config": "absent.yaml", "seeds": [0]},
                            {"case_id": "healthy", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "BLOCKED"
    assert report["cases"][0]["process"] is None
    assert report["cases"][0]["error"]["type"] == "FileNotFoundError"
    assert report["cases"][0]["requested_repeats"] is None
    assert report["cases"][1]["status"] == "COMPLETED"
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_preexisting_stop_does_not_launch_a_process(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    monkeypatch.setattr(module, "run_owned_process", lambda *a, **kw: pytest.fail("must not launch after stop"))
    runner = ExperimentSuiteRunner(suite)
    assert runner.stop() and not runner.stop()
    report = runner.evaluate()
    assert report["status"] == "STOPPED" and report["summary"] is None
    assert report["population"]["outcome_counts"] == {"NOT_RUN": 2}
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_zero_exit_without_run_evidence_is_failure(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    def empty(command, cwd, environment, timeout_s, log_path, **kwargs):
        return run_owned_process([sys.executable, "-c", "pass"], cwd, environment, timeout_s, log_path, **kwargs)
    monkeypatch.setattr(module, "run_owned_process", empty)
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "FAIL" and report["summary"] is None
    assert all(case["error"]["type"] == "MissingExperimentResult" for case in report["cases"])
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_hung_process_is_timed_out_without_claiming_a_trial_result(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    configure(suite, process_timeout_s=1., fail_fast=True)
    def hung(command, cwd, environment, timeout_s, log_path, **kwargs):
        return run_owned_process([sys.executable, "-S", "-c", "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(30)"],
                                 cwd, environment, timeout_s, log_path, **kwargs)
    monkeypatch.setattr(module, "run_owned_process", hung)
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "FAIL" and report["cases"][0]["status"] == "TIMED_OUT"
    assert report["cases"][0]["process"]["root_reaped"]
    assert report["cases"][0]["process"]["termination_signals"][:2] == ["SIGINT", "SIGTERM"]
    assert report["cases"][1]["status"] == "NOT_RUN"
    assert report["summary"] is None
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_thread_stop_terminates_owned_case_and_retains_remaining_population(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    launched = threading.Event()
    def hanging(command, cwd, environment, timeout_s, log_path, **kwargs):
        original = kwargs["on_started"]
        def started(pid):
            original(pid)
            launched.set()
        kwargs["on_started"] = started
        return run_owned_process([sys.executable, "-c", "import time; time.sleep(30)"], cwd, environment, timeout_s, log_path, **kwargs)
    monkeypatch.setattr(module, "run_owned_process", hanging)
    runner, results = ExperimentSuiteRunner(suite), []
    thread = threading.Thread(target=lambda: results.append(runner.evaluate()))
    thread.start()
    assert launched.wait(5)
    with pytest.raises(RuntimeError, match="active evaluation"):
        runner.evaluate()
    assert runner.stop()
    thread.join(5)
    assert not thread.is_alive()
    report = results[0]
    assert report["status"] == "STOPPED"
    assert [case["status"] for case in report["cases"]] == ["STOPPED", "NOT_RUN"]
    assert report["cases"][0]["process"]["root_reaped"]
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_unreaped_root_prevents_next_case_even_when_child_evidence_is_invalid(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    calls = []
    def unreaped(command, cwd, environment, timeout_s, log_path, **kwargs):
        calls.append(command)
        kwargs["on_started"](123)
        log_path.write_text("synthetic unreaped process receipt for a scheduling fault test")
        result = log_path.parent / "artifacts/bad/results.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}")
        return {"pid": 123, "exit_code": None, "root_reaped": False, "stop_reason": "process_timeout", "timed_out": True,
                "timeout_s": timeout_s, "cleanup_grace_s": kwargs["grace_s"], "termination_signals": ["SIGINT", "SIGTERM", "SIGKILL"]}
    monkeypatch.setattr(module, "run_owned_process", unreaped)
    report = ExperimentSuiteRunner(suite).evaluate()
    assert len(calls) == 1 and report["status"] == "FAIL"
    assert report["cases"][1]["status"] == "NOT_RUN"
    assert report["cases"][1]["reason"] == "process_cleanup_pending"


def test_input_changes_stop_scheduling_and_keep_the_already_completed_run(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    def edit_after_first(*args, **kwargs):
        receipt = run_owned_process(*args, **kwargs)
        original = suite.parent / "fixture/experiment.yaml"
        original.write_text(original.read_text() + "\n# changed after first case\n")
        return receipt
    monkeypatch.setattr(module, "run_owned_process", edit_after_first)
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "FAIL" and not report["source_stable"]
    assert [case["status"] for case in report["cases"]] == ["COMPLETED", "NOT_RUN"]
    assert report["summary"]["population"]["valid_runs"] == 1
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_result_for_a_different_seed_cannot_enter_the_suite_population(suite, monkeypatch):
    from xpolicy.evaluation import suite as module
    configure(suite, fail_fast=True)
    def different_seed(command, cwd, environment, timeout_s, log_path, **kwargs):
        path = Path(command[-1])
        config = yaml.safe_load(path.read_text())
        config["experiment"]["seed"] = 999
        alternate = path.with_name("wrong-seed.yaml")
        alternate.write_text(yaml.safe_dump(config))
        return run_owned_process(command[:-1] + [str(alternate)], cwd, environment, timeout_s, log_path, **kwargs)
    monkeypatch.setattr(module, "run_owned_process", different_seed)
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "FAIL" and report["summary"] is None
    assert "planned case/seed" in report["cases"][0]["error"]["message"]
    assert report["cases"][1]["status"] == "NOT_RUN"


@pytest.mark.parametrize("mutation", ["log", "summary", "seed", "status", "population"])
def test_suite_verifier_rejects_changed_evidence_and_metadata(suite, mutation):
    configure(suite, cases=[{"case_id": "one", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1}])
    report = ExperimentSuiteRunner(suite).evaluate()
    path = suite.parent / "suite-results/SUITE_REPORT.json"
    if mutation == "log":
        log = path.parent / report["cases"][0]["directory"] / "process.log"
        log.write_text(log.read_text() + "changed")
    else:
        if mutation == "summary":
            report["summary"]["population"]["valid_runs"] += 1
        elif mutation == "seed":
            report["cases"][0]["seed"] = 999
        elif mutation == "status":
            report["cases"][0]["status"] = "NOT_RUN"
        else:
            report["population"]["planned_cases"] += 1
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_suite(path)


@pytest.mark.parametrize("change", [{"cases": []}, {"process_timeout_s": float("nan")}, {"fail_fast": "false"},
                                    {"cleanup_grace_s": 0.}, {"cases": [{"case_id": "../escape", "config": "x", "seeds": [0]}]},
                                    {"cases": [{"case_id": "one", "config": "x", "seeds": [True]}]},
                                    {"cases": [{"case_id": "one", "config": "x", "seeds": [0, 0]}]}])
def test_suite_schema_rejects_ambiguous_or_invalid_plans(change):
    config = {"name": "test", "artifacts_root": "results", "cases": [{"case_id": "one", "config": "x", "seeds": [0]}], **change}
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(config)


def test_cli_suite_and_verification(suite):
    result = subprocess.run([sys.executable, "-m", "xpolicy", "eval", "suite", "--config", str(suite)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"
    path = suite.parent / "suite-results/SUITE_REPORT.json"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "results", "verify-suite", "--report", str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "xpolicy", "eval", "suite", "--config", str(suite)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "File exists" in result.stderr


def test_fixture_generator_produces_an_explicit_two_strategy_suite(tmp_path):
    experiment = create_fixture(tmp_path / "demo", frames=6, with_suite=True)
    path = experiment.parent / "suite.yaml"
    settings = SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
    assert [(case.case_id, case.seeds, case.repeats) for case in settings.cases] == [
        ("sync", (3, 7), 1), ("async_latest_only", (3, 7), 1)]
    assert yaml.safe_load((experiment.parent / "alternate.yaml").read_text())["execution"]["latency_trim"] is True


def test_explicit_interpreter_retains_venv_prefix_and_actual_runtime_inventory(suite):
    configure(suite, cases=[{"case_id": "selected", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1,
                            "python": sys.executable}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "PASS"
    case = report["cases"][0]
    assert case["process"]["command"][0] == sys.executable
    inventory = json.loads((suite.parent / "suite-results" / case["result"]).with_name("software-inventory.json").read_text())
    assert inventory["prefix"] == sys.prefix and inventory["executable"] == sys.executable
    assert inventory["python"] == sys.version
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_missing_selected_interpreter_retains_a_blocked_case(suite):
    configure(suite, cases=[{"case_id": "absent", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1,
                            "python": "missing-env/bin/python"}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "BLOCKED"
    assert report["cases"][0]["status"] == "BLOCKED"
    assert report["cases"][0]["reason"] == "process_start_failed"
    assert report["cases"][0]["process_pid"] is None and report["cases"][0]["requested_repeats"] == 1
    assert verify_suite(suite.parent / "suite-results/SUITE_REPORT.json") == report


def test_rehashed_plan_cannot_change_the_declared_seed_and_effective_configuration(suite):
    from xpolicy.core.artifacts import file_sha256
    from xpolicy.evaluation.suite import _artifact_hashes
    configure(suite, cases=[{"case_id": "one", "config": "fixture/experiment.yaml", "seeds": [3], "repeats": 1}])
    runner = ExperimentSuiteRunner(suite)
    runner.stop()
    report = runner.evaluate()
    root = suite.parent / "suite-results"
    plan = json.loads((root / "plan.json").read_text())
    entry = plan["cases"][0]
    entry["expected_config"]["experiment"]["seed"] = 999
    (root / entry["config"]).write_text(yaml.safe_dump(entry["expected_config"]))
    entry["configuration_sha256"] = file_sha256(root / entry["config"])
    (root / "plan.json").write_text(json.dumps(plan))
    report["plan_sha256"] = file_sha256(root / "plan.json")
    report["evidence_sha256"] = _artifact_hashes(root)
    (root / "SUITE_REPORT.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="overrides differ"):
        verify_suite(root / "SUITE_REPORT.json")


def test_closed_suite_verifies_after_original_inputs_and_config_symlink_are_removed(suite):
    link = suite.parent / "config-link"
    link.symlink_to(suite.parent / "fixture", target_is_directory=True)
    configure(suite, cases=[{"case_id": "linked", "config": "config-link/experiment.yaml", "seeds": [3], "repeats": 1}])
    report = ExperimentSuiteRunner(suite).evaluate()
    assert report["status"] == "PASS"
    link.unlink()
    shutil.rmtree(suite.parent / "fixture")
    suite.unlink()
    moved = suite.parent / "portable-evidence"
    shutil.move(suite.parent / "suite-results", moved)
    assert verify_suite(moved / "SUITE_REPORT.json") == report


@pytest.mark.parametrize("body", ["[]", "{}", '{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'])
def test_invalid_suite_json_is_rejected_without_a_traceback(tmp_path, body):
    path = tmp_path / "SUITE_REPORT.json"
    path.write_text(body)
    with pytest.raises(ValueError):
        verify_suite(path)
    result = subprocess.run([sys.executable, "-m", "xpolicy", "results", "verify-suite", "--report", str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "Traceback" not in result.stderr


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
def test_cli_interrupt_forwards_stop_to_active_runner_and_retains_unstarted_case(suite, signal_name):
    import signal
    # A long paced replay supplies a real interruptible Runner, without an SDK.
    experiment = create_fixture(suite.parent / "long", frames=300, strategy="async_latest_only")
    configure(suite, cleanup_grace_s=2., cases=[{"case_id": "long", "config": str(experiment), "seeds": [3, 7], "repeats": 1}])
    process = subprocess.Popen([sys.executable, "-m", "xpolicy", "eval", "suite", "--config", str(suite)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    root = suite.parent / "suite-results"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            traces = list(root.glob("cases/long/seed-3/artifacts/*/trace.jsonl"))
            if traces and '"offline_readiness"' in traces[0].read_text():
                break
            assert process.poll() is None
            time.sleep(.02)
        else:
            pytest.fail("suite child did not reach actual Runner readiness")
        process.send_signal(getattr(signal, signal_name))
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 2, stderr
        report = json.loads(stdout)
        assert report["status"] == "STOPPED"
        assert [case["status"] for case in report["cases"]] == ["STOPPED", "NOT_RUN"]
        assert report["cases"][0]["run_outcome"] == "STOPPED"
        assert report["cases"][0]["process"]["root_reaped"]
        assert verify_suite(root / "SUITE_REPORT.json") == report
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=10)
