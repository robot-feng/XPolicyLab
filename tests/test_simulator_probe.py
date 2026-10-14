import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from tools.probe_simulator_runtime import native_constraints, probe, run_child


def no_site_python(tmp_path):
    path = tmp_path / "python without SDK packages"
    path.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' -S "$@"\n')
    path.chmod(0o755)
    return path


def test_package_probe_is_supervised_and_retains_actual_interpreter_metadata(tmp_path):
    report = probe(Path(sys.executable), "packages", tmp_path / "packages", timeout_s=10)
    assert report["status"] == "PASS" and report["source_stable"]
    assert report["child"]["mode"] == "packages"
    assert report["child"]["executable"] == sys.executable
    assert report["execution"]["exit_code"] == 0
    assert set(report["evidence_sha256"]) >= {"child.json", "runtime.log"}
    assert report["execution"]["root_reaped"]
    assert report["provenance"]["probe_source_sha256"] == report["evidence_sha256"]["probe_child.py"]
    assert "benchmark_task" in report["not_verified"]
    assert (tmp_path / "packages/native-stack-constraints.txt").is_file()


def test_missing_simulator_package_is_blocked_without_replacing_the_engine(tmp_path):
    report = probe(no_site_python(tmp_path), "sapien_cpu", tmp_path / "missing-sdk", timeout_s=10)
    assert report["status"] == "BLOCKED"
    assert report["child"]["error"]["type"] == "ModuleNotFoundError"
    assert not any(item["name"] == "actual_CPU_physics_steps" for item in report["child"]["stages"])


def test_missing_interpreter_keeps_report_and_does_not_overwrite_previous_evidence(tmp_path):
    output = tmp_path / "absent"
    report = probe(tmp_path / "missing-python", "packages", output, timeout_s=10)
    assert report["status"] == "BLOCKED"
    assert json.loads((output / "SIMULATOR_PROBE_REPORT.json").read_text())["error"]["type"] == "InterpreterMissing"
    with pytest.raises(FileExistsError):
        probe(Path(sys.executable), "packages", output, timeout_s=10)


def test_timeout_terminates_owned_process_group_and_keeps_nonzero_native_exit(tmp_path):
    result = run_child([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, dict(os.environ), .05, tmp_path / "timeout.log")
    assert result["timed_out"] and result["exit_code"] != 0 and result["duration_s"] < 8


@pytest.mark.skipif(sys.platform != "linux", reason="inspects the Linux owned worker process state")
def test_timeout_kills_worker_that_ignores_term_after_root_exits(tmp_path):
    import time
    worker_code = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open('worker.pid','w').write(str(os.getpid())); time.sleep(30)"
    root_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(worker_code) + "]); time.sleep(30)"
    result = run_child([sys.executable, "-c", root_code], tmp_path, dict(os.environ), 1., tmp_path / "worker-timeout.log")
    assert result["timed_out"] and result["root_reaped"]
    assert result["termination_signals"] == ["SIGTERM", "SIGKILL"]
    worker = int((tmp_path / "worker.pid").read_text())
    for _ in range(100):
        stat = Path(f"/proc/{worker}/stat")
        if not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z"):
            break
        time.sleep(.01)
    else:
        os.kill(worker, 9)
        pytest.fail("owned worker remained alive after supervisor timeout")


def test_native_failure_is_not_reported_as_a_timeout_or_pass(tmp_path):
    result = run_child([sys.executable, "-c", "raise SystemExit(139)"], tmp_path, dict(os.environ), 10, tmp_path / "native-exit.log")
    assert result["exit_code"] == 139 and not result["timed_out"]


def test_zero_exit_without_child_evidence_fails(tmp_path, monkeypatch):
    from tools import probe_simulator_runtime as module
    monkeypatch.setattr(module, "run_child", lambda *a: {"exit_code": 0, "timed_out": False})
    report = probe(Path(sys.executable), "packages", tmp_path / "no-evidence", timeout_s=10)
    assert report["status"] == "FAIL" and report["error"]["type"] == "InvalidProbeEvidence"


@pytest.mark.parametrize("kwargs", [{"mode": "fake"}, {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
                                    {"timeout_s": 0}, {"gpu": -1}, {"gpu": True}])
def test_invalid_parameters_do_not_create_a_partial_run(tmp_path, kwargs):
    arguments = {"python": Path(sys.executable), "mode": "packages", "output": tmp_path / "invalid", **kwargs}
    with pytest.raises(ValueError):
        probe(**arguments)
    assert not arguments["output"].exists()


@pytest.mark.parametrize("evidence", [[], {}, {"status": "PASS", "mode": "packages", "scope": "unrelated"}])
def test_malformed_child_evidence_is_retained_as_failure(tmp_path, monkeypatch, evidence):
    from tools import probe_simulator_runtime as module
    output = tmp_path / "malformed"
    def fake_child(*args):
        (output / "child.json").write_text(json.dumps(evidence))
        return {"exit_code": 0, "timed_out": False}
    monkeypatch.setattr(module, "run_child", fake_child)
    report = probe(Path(sys.executable), "packages", output)
    assert report["status"] == "FAIL"
    assert report["child"] == evidence
    assert json.loads((output / "SIMULATOR_PROBE_REPORT.json").read_text())["status"] == "FAIL"


def test_supervisor_interruption_keeps_terminal_evidence(tmp_path, monkeypatch):
    from tools import probe_simulator_runtime as module
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "run_child", interrupted)
    report = probe(Path(sys.executable), "packages", tmp_path / "interrupted")
    assert report["status"] == "BLOCKED" and report["source_stable"]
    assert report["error"]["type"] == "SupervisorInterrupted"


def test_native_signal_survives_without_completed_child_report(tmp_path, monkeypatch):
    from tools import probe_simulator_runtime as module
    monkeypatch.setattr(module, "run_child", lambda *a: {"exit_code": -11, "timed_out": False})
    report = probe(Path(sys.executable), "sapien_render", tmp_path / "signal")
    assert report["status"] == "FAIL"
    assert report["error"]["signal"] == "SIGSEGV" and report["error"]["exit_code"] == -11


def test_native_constraints_keep_local_cuda_versions_but_do_not_pin_common_core_packages():
    from types import SimpleNamespace
    def package(name, version):
        return SimpleNamespace(metadata={"Name": name}, version=version)
    distributions = [package("torch", "2.7.0+cu128"), package("NVIDIA_CUDA_RUNTIME_CU12", "12.8.57"),
                     package("numpy", "1.26.0"), package("isaaclab-assets", "0.2.4"), package("websockets", "12.0")]
    assert native_constraints(distributions) == ["isaaclab-assets==0.2.4", "numpy==1.26.0",
                                                "nvidia-cuda-runtime-cu12==12.8.57", "torch==2.7.0+cu128"]
    with pytest.raises(ValueError, match="conflicting installed versions"):
        native_constraints(distributions + [package("NumPy", "2.2.6")])
    with pytest.raises(ValueError, match="invalid installed version"):
        native_constraints([package("torch", "2.7.0\nmalformed")])


def test_probe_never_sets_vendor_license_acceptance(tmp_path, monkeypatch):
    from tools import probe_simulator_runtime as module
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    def inspect_child(command, cwd, environment, *args):
        assert "OMNI_KIT_ACCEPT_EULA" not in environment
        assert (cwd / "probe_child.py").is_file()
        assert command[2] == str(cwd / "probe_child.py")
        return {"exit_code": 0, "timed_out": False}
    monkeypatch.setattr(module, "run_child", inspect_child)
    report = probe(Path(sys.executable), "isaac_app", tmp_path / "license")
    assert report["inherited_runtime_flags"]["OMNI_KIT_ACCEPT_EULA"] is None
    assert report["status"] == "FAIL"  # No successful SDK evidence was produced.
