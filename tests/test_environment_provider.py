from dataclasses import asdict, replace
import json
import subprocess
import sys

import numpy as np
import pytest

from tools.create_mock_dataset import create_fixture
from xpolicy.core.config import load_experiment
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.registry import Entry
from xpolicy.environment.contracts import EpisodeContext
from xpolicy.environment.registry import EnvironmentEntry, EnvironmentRegistry, environment_registry
from xpolicy.evaluation.runner import _preflight, run_experiment
from xpolicy.execution.buffer import ActionBuffer
from xpolicy.policy.runtime import build_policy


@pytest.fixture
def environment(tmp_path):
    path = create_fixture(tmp_path / "recorded", frames=5)
    config, contracts = load_experiment(path)
    provider = environment_registry().create("dataset_direct", config, contracts, _preflight(config, contracts))
    try:
        yield path, config, contracts, provider
    finally:
        provider.close()


def commands(config, contracts, provider, episode="one"):
    buffer = ActionBuffer(contracts.manifest, config.execution)
    generation = buffer.reset(episode)
    provider.reset(EpisodeContext(episode, generation, 42, 0))
    obs = provider.observe(0)
    request = buffer.issue(obs, control_tick=0)
    model = build_policy(contracts)
    try:
        model.reset_episode(episode)
        prediction = model.predict_action_chunk(obs)
    finally:
        model.close()
    assert buffer.accept(request, prediction, control_tick=0, now_s=obs.timestamp_s, clock_domain=obs.clock_domain)
    return [buffer.take(i, now_s=provider.time_at(i)[1], clock_domain=obs.clock_domain) for i in range(2)]


def test_registry_metadata_is_lazy_and_unavailable_modes_never_load_factory(monkeypatch):
    code = "import sys; from xpolicy.environment.registry import environment_registry; r=environment_registry(); assert len(r.entries()) == 7; assert not any(n.split('.')[0] in {'h5py','torch','transformers','rospy','isaaclab'} for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    def no_load(self):
        raise AssertionError("unavailable environment must be rejected before import")
    monkeypatch.setattr(Entry, "load", no_load)
    registry = environment_registry()
    assert registry.get("dataset_direct").status == "IMPLEMENTED"
    for name in ("dataset_wire", "fake_plant", "sdk_mock", "physics_sim", "shadow", "real"):
        with pytest.raises(UnsupportedCapability, match=name):
            registry.create(name)


@pytest.mark.parametrize("entry", [Entry("wrong", "p:Factory"), EnvironmentEntry("wrong", "", status="IMPLEMENTED"),
                                    EnvironmentEntry("wrong", "p:Factory", status="DEFERRED", reason="pending"),
                                    EnvironmentEntry("wrong", "", status="NOT_IMPLEMENTED"),
                                    EnvironmentEntry("wrong", "", status="guessed")])
def test_registry_rejects_incoherent_implementation_claim(entry):
    with pytest.raises(ValueError):
        EnvironmentRegistry().register(entry)


def test_recorded_session_owns_generation_and_cannot_change_recorded_observations(environment):
    _, config, contracts, provider = environment
    with pytest.raises(RuntimeError, match="active reset"):
        provider.observe(0)
    scheduled = commands(config, contracts, provider)
    before = provider.observe(0)
    altered = [replace(command, values=np.full(2, 100., dtype=np.float32)) for command in scheduled]
    receipt = provider.record(altered, control_tick=0)
    assert not receipt.values.flags.writeable and not receipt.labels.flags.writeable
    assert receipt.causal is False and receipt.device_feedback is None
    assert receipt.control_frame_ids == (0, 1)
    assert receipt.control_timestamps_s == (0., 1. / contracts.dataset.sample_hz)
    after = provider.observe(0)
    np.testing.assert_array_equal(before.state, after.state)
    np.testing.assert_array_equal(before.images["cam_head"], after.images["cam_head"])
    with pytest.raises(ValueError, match="duplicate"):
        provider.record(scheduled, control_tick=0)
    assert provider.stop("done") and not provider.stop("done")
    with pytest.raises(RuntimeError, match="active reset"):
        provider.record(scheduled, control_tick=0)
    reset = provider.reset(EpisodeContext("two", 2, 42, 1))
    assert reset["physical_robot_reset"] is reset["device_ack"] is None
    with pytest.raises(ValueError, match="episode/generation"):
        provider.record(scheduled, control_tick=0)
    assert provider.verify_evidence() == config.environment.dataset_path


def test_submission_validates_timing_identity_and_shape_before_sink_mutates(environment):
    _, config, contracts, provider = environment
    scheduled = commands(config, contracts, provider)
    bad = replace(scheduled[0], ticket=replace(scheduled[0].ticket, source_frame_id=99))
    with pytest.raises(ValueError, match="source identity"):
        provider.record([bad], control_tick=0)
    with pytest.raises(ValueError, match="contiguous"):
        provider.record([replace(scheduled[0], control_tick=3)], control_tick=0)
    with pytest.raises(ValueError, match="finite action layout"):
        provider.record([replace(scheduled[0], values=np.zeros(3))], control_tick=0)
    with pytest.raises(IndexError, match="outside"):
        provider.time_at(5)
    assert provider.record(scheduled, control_tick=0).values.shape == (2, 2)
    with pytest.raises(ValueError, match="must increase"):
        provider.reset(EpisodeContext("one", 1, 42, 0))
    provider.close(); provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.reset(EpisodeContext("two", 2, 42, 1))


@pytest.mark.parametrize("changes", [{"generation": True}, {"generation": 0}, {"seed": -1}, {"seed": True}, {"repeat_index": -1}, {"episode_id": " "}])
def test_episode_context_is_explicit_and_strict(changes):
    with pytest.raises(ValueError):
        EpisodeContext(**({"episode_id": "one", "generation": 1, "seed": 0, "repeat_index": 0} | changes))


def test_runner_retains_provider_preparation_reset_and_stops_each_repeat(environment):
    path, _, _, _ = environment
    report = run_experiment(path)
    assert report["status"] == "COMPLETED"
    assert report["environment_provider"]["runtime_contract"] == "recorded_timeline_v1"
    assert report["environment_provider"]["causal_feedback"] is False
    from pathlib import Path
    events = [json.loads(line) for line in (Path(report["artifacts"]) / "trace.jsonl").read_text().splitlines()]
    resets = [event["receipt"] for event in events if event["stage"] == "environment_reset"]
    assert [item["generation"] for item in resets] == [1, 2]
    from xpolicy.evaluation.reports import summarize_reports
    result_path = Path(report["artifacts"]) / "results.json"
    assert summarize_reports([result_path])["status"] == "PASS"
    report["environment_provider"]["causal_feedback"] = True
    result_path.write_text(json.dumps(report))
    assert summarize_reports([result_path])["status"] == "FAIL"


def test_environment_list_cli_discloses_deferred_capabilities():
    result = subprocess.run([sys.executable, "-m", "xpolicy", "environments", "list", "--json"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    entries = {item["name"]: item for item in json.loads(result.stdout)}
    assert entries["dataset_direct"]["runtime_contract"] == "recorded_timeline_v1"
    assert entries["physics_sim"]["status"] == "NOT_IMPLEMENTED"
    assert entries["real"]["factory"] == "" and entries["real"]["reason"]


def test_frame_ids_are_not_assumed_to_be_control_ticks(tmp_path):
    import h5py
    import yaml
    from xpolicy.core.artifacts import file_sha256
    path = create_fixture(tmp_path / "nonconsecutive", frames=5)
    data = path.parent / "trajectory.hdf5"
    with h5py.File(data, "r+") as file:
        file["frame_ids"][:] = [10, 20, 31, 45, 70]
    profile_path = path.parent / "dataset.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["artifact"]["sha256"] = file_sha256(data)
    profile_path.write_text(yaml.safe_dump(profile))
    config, contracts = load_experiment(path)
    source = environment_registry().create("dataset_direct", config, contracts, _preflight(config, contracts))
    try:
        scheduled = commands(config, contracts, source)
        assert source.record(scheduled, control_tick=0).control_frame_ids == (10, 20)
        assert source.observe(2).frame_id == 31
    finally:
        source.close()


def test_environment_doctor_checks_actual_provider_without_policy_loading(environment, monkeypatch):
    from xpolicy.diagnostics.doctor import doctor
    from xpolicy.evaluation import runner
    path, _, _, _ = environment
    monkeypatch.setattr(runner, "build_policy", lambda *args: pytest.fail("provider doctor must not construct a policy"))
    report = doctor(path, mode="environment")
    assert report["status"] == "PASS" and report["scope"] == "environment_provider_preflight"
    item = next(item for item in report["checks"] if item["id"] == "environment_provider_preflight")
    assert item["status"] == "PASS" and item["details"]["frames"] == 5
    assert next(item for item in report["checks"] if item["id"] == "environment_runtime")["status"] == "NOT_RUN"


def test_contract_doctor_does_not_claim_deferred_environment_is_available(environment):
    import yaml
    from xpolicy.diagnostics.doctor import doctor
    path, _, _, _ = environment
    config = yaml.safe_load(path.read_text())
    config["environment"]["kind"] = "fake_plant"
    path.write_text(yaml.safe_dump(config))
    report = doctor(path)
    assert report["status"] == "PASS" and report["scope"] == "contract_preflight"
    check = next(item for item in report["checks"] if item["id"] == "environment_provider")
    assert check["status"] == "NOT_RUN" and check["details"]["status"] == "DEFERRED"
    assert doctor(path, mode="environment")["status"] == "BLOCKED"
