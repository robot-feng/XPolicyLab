import json
import select
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from xpolicy.core.artifacts import artifact_sha256, file_sha256, verify_artifact, write_json
from xpolicy.core.config import ServerSettings, ValidationConfig, load_experiment, load_service, read_yaml
from xpolicy.core.contracts import ArtifactIdentity, ModelManifest
from xpolicy.core.registry import Entry, Registry, policy_registry
from xpolicy.diagnostics.doctor import doctor
from xpolicy.policy.runtime import LegacyPolicyShim, build_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text", ["policy: mock\npolicy: StarVLA\n", "policy: [\n", "- item\n"])
def test_bad_yaml(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_yaml(path)


def test_paths_resolve_from_declaring_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config, contracts = load_experiment(ROOT / "configs/experiments/mock_contract.yaml")
    assert contracts.manifest_path == ROOT / "configs/policies/mock_manifest.yaml"
    assert config.environment.robot == ROOT / "configs/robots/mock_arm.yaml"
    assert config.artifacts.root == ROOT / "runs/mock_contract_baseline"


@pytest.mark.parametrize("value", [1, "true", False, None])
def test_cannot_disable_contracts(value):
    with pytest.raises(ValidationError):
        ValidationConfig(strict_contract=value)


@pytest.mark.parametrize("change", [{"port": True}, {"port": -1}, {"port": 65536}, {"ws_ping_interval_s": True}, {"ws_ping_timeout_s": float("nan")}])
def test_server_strict_fields(change):
    with pytest.raises(ValidationError):
        ServerSettings(**change)


def test_registry_is_lazy():
    result = subprocess.run([sys.executable, "-c", "import sys; from xpolicy.core.registry import policy_registry; r = policy_registry(); assert r.get('mock'); assert not any(n.split('.')[0] in {'torch', 'transformers', 'rospy', 'gdk'} for n in sys.modules); assert not any('.policy.' in n and n.endswith('.model') for n in sys.modules)"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    registry = Registry("test")
    entry = Entry("sample", "missing_package:Model")
    registry.register(entry)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(entry)
    with pytest.raises(ValueError, match="unknown"):
        registry.get("unknown")


def test_cli_and_contract_scope(tmp_path):
    for args in (["--help"], ["policy", "list", "--json"]):
        result = subprocess.run([sys.executable, "-m", "xpolicy", *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    output = tmp_path / "doctor.json"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "doctor", "--config", str(ROOT / "configs/experiments/mock_contract.yaml"), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["scope"] == "contract_preflight"
    assert report["status"] == "PASS"
    assert next(item for item in report["checks"] if item["id"] == "environment_runtime")["status"] == "NOT_RUN"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "doctor", "--config", str(tmp_path / "missing.yaml")], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_cli_serve_starts_and_shuts_down():
    process = subprocess.Popen([sys.executable, "-m", "xpolicy", "policy", "serve", "--config", str(ROOT / "configs/policies/mock.yaml"), "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 8)[0], "CLI did not announce readiness"
        event = json.loads(process.stdout.readline())
        assert event["event"] == "policy_server_ready"
        assert event["api"] == "chunk"
        assert event["url"].startswith("ws://127.0.0.1:")
        assert not event["url"].endswith(":0")
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_legacy_shim_preserves_result_and_blocks_chunk():
    result = [{"joint_state": [1, 2]}]
    class Legacy:
        def get_action(self): return result
        def predict_action_chunk(self, obs): return "must not be accessible"
        def reset(self): return "original_reset"
    shim = LegacyPolicyShim(Legacy())
    assert shim.get_action() is result
    assert shim.reset() == "original_reset"
    assert not hasattr(shim, "predict_action_chunk")


def test_artifact_content_and_declaring_path(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"weights")
    identity = ArtifactIdentity(uri="weights.bin", sha256=file_sha256(path))
    assert verify_artifact(identity, tmp_path / "profile.yaml") == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_artifact(identity, tmp_path / "profile.yaml")
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    weights = directory / "weights.bin"
    weights.write_bytes(b"weights")
    expected = artifact_sha256(directory)
    assert expected == artifact_sha256(directory)
    (directory / "empty").mkdir()
    assert expected == artifact_sha256(directory)
    weights.write_bytes(b"changed")
    assert expected != artifact_sha256(directory)
    (directory / "link").symlink_to(weights)
    with pytest.raises(ValueError, match="symlink"):
        artifact_sha256(directory)


def test_bad_artifact_rejected_before_model_import(contracts, tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    manifest = ModelManifest.model_validate({**contracts.manifest.model_dump(), "source_kind": "checkpoint", "checkpoint": {"uri": str(weights), "sha256": "0" * 64}, "training_loader": {"uri": str(weights), "sha256": file_sha256(weights)}})
    registry = Registry("policy")
    registry.register(Entry("vendor", "nonexistent_vendor:Model", ("chunk",)))
    loaded = replace(contracts, policy=contracts.policy.model_copy(update={"name": "vendor"}), manifest=manifest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_policy(loaded, registry)


def test_atomic_report_rejects_nan(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"status": "PASS"})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"status": "PASS"}
    assert list(tmp_path.iterdir()) == [path]
