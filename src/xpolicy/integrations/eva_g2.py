"""Safe loopback smoke bridge from EVA's G2 Fake node to an XPL policy service.

This module intentionally supports the EVA software Fake node only. The real G2
node has unresolved image-channel and gripper-name differences and is not a
valid target for this action publisher.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from xpolicy.core.artifacts import file_sha256, source_revision, write_json
from xpolicy.core.config import ExecutionConfig, LoadedContracts, load_service
from xpolicy.core.contracts import ContractValidator, Observation, RobotProfile
from xpolicy.execution.async_latest import AsyncLatestOnly, WebSocketPolicyBackend

_INVENTORY_PACKAGES = {
    "numpy", "torch", "torchvision", "opencv-python", "opencv-python-headless",
    "pyzmq", "websockets", "pydantic", "msgpack", "msgpack-numpy",
    "openpi-client", "eva-client", "agibot-gdk", "jax", "jaxlib", "pytest",
}


def _eva_msgpack_numpy():
    # EVA imports this implementation; the standalone package uses another array extension.
    from openpi_client import msgpack_numpy

    return msgpack_numpy


@dataclass(frozen=True)
class EvaWireObservation:
    """The source-side EVA ZMQ frame, before conversion into the XPL contract."""

    timestamp_s: float
    images: Mapping[str, np.ndarray]
    state: Mapping[str, np.ndarray]


def decode_eva_observation(payload: bytes) -> EvaWireObservation:
    """Decode EVA's msgpack-numpy observation shape without channel conversion."""
    raw = _eva_msgpack_numpy().unpackb(payload)
    if not isinstance(raw, Mapping):
        raise ValueError("EVA observation payload must be a mapping")
    if "t" not in raw or "images" not in raw or "state" not in raw:
        raise ValueError("EVA observation requires t, images and state")
    timestamp = raw["t"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.number)):
        raise ValueError("EVA observation timestamp must be numeric")
    timestamp = float(timestamp)
    if not np.isfinite(timestamp):
        raise ValueError("EVA observation timestamp must be finite")
    if not isinstance(raw["images"], Mapping) or not isinstance(raw["state"], Mapping):
        raise ValueError("EVA images and state must be mappings")

    images = {}
    for name, value in raw["images"].items():
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not image.size:
            raise ValueError(f"EVA image {name!r} must be non-empty HWC uint8")
        images[str(name)] = image

    state = {}
    for name, value in raw["state"].items():
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"EVA state group {name!r} must be a finite vector")
        state[str(name)] = vector
    return EvaWireObservation(timestamp, images, state)


def to_xpl_observation(
    wire: EvaWireObservation,
    robot: RobotProfile,
    *,
    frame_id: int,
    episode_id: str,
) -> Observation:
    """Map EVA group arrays by declared names; reject implicit dimension matches."""
    images: dict[str, np.ndarray] = {}
    for camera in robot.cameras:
        source_key = camera.source_key or camera.name
        if source_key not in wire.images:
            raise ValueError(f"EVA frame is missing configured camera source {source_key!r}")
        image = wire.images[source_key]
        if camera.shape is not None and image.shape != camera.shape:
            raise ValueError(
                f"EVA camera {source_key!r} has shape {image.shape}, expected {camera.shape}"
            )
        # EVA Fake G2 emits RGB arrays. Preserve those bytes unchanged.
        images[camera.name] = image

    expected_groups = set(robot.legacy_state_groups)
    if set(wire.state) != expected_groups:
        raise ValueError(
            f"EVA state groups differ from the robot profile: "
            f"missing={sorted(expected_groups-set(wire.state))}, "
            f"extra={sorted(set(wire.state)-expected_groups)}"
        )
    by_name: dict[str, float] = {}
    for group, names in robot.legacy_state_groups.items():
        values = wire.state[group]
        if values.shape != (len(names),):
            raise ValueError(
                f"EVA state group {group!r} has shape {values.shape}; expected ({len(names)},)"
            )
        by_name.update((name, float(value)) for name, value in zip(names, values, strict=True))

    state_names = tuple(field.name for field in robot.state_fields)
    if set(by_name) != set(state_names):
        raise ValueError("EVA group-to-joint mapping does not cover the declared state fields")
    return Observation(
        images=images,
        state=np.asarray([by_name[name] for name in state_names], dtype=np.float32),
        state_names=state_names,
        state_units=tuple(field.unit for field in robot.state_fields),
        instruction="EVA G2 Fake integration smoke",
        timestamp_s=wire.timestamp_s,
        clock_domain="eva_host_monotonic",
        frame_id=frame_id,
        episode_id=episode_id,
    )


def encode_fake_action(values: np.ndarray) -> bytes:
    """Encode one XPL scheduled action for the isolated EVA Fake node.

    The EVA fake wire uses target="real" even though this harness connects only
    to a child process started from fake_node.py on loopback. There is no external
    endpoint option and this function must not be reused for a device client.
    """
    action = np.asarray(values, dtype=np.float32)
    if action.shape != (24,) or not np.isfinite(action).all():
        raise ValueError("fake G2 action must be a finite 24D vector")
    return _eva_msgpack_numpy().packb({"t": time.monotonic(), "action": action, "target": "real"})


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _packages() -> list[dict[str, str]]:
    return sorted(
        [
            {"name": item.metadata.get("Name", "unknown"), "version": item.version}
            for item in importlib.metadata.distributions()
            if item.metadata.get("Name", "").lower() in _INVENTORY_PACKAGES
        ],
        key=lambda item: item["name"].lower(),
    )


def _eva_inventory(python: Path) -> dict[str, Any]:
    selected = sorted(_INVENTORY_PACKAGES)
    code = (
        "import importlib.metadata as m, json, platform, sys; "
        "print(json.dumps({'python': sys.version, 'executable': sys.executable, "
        f"'platform': platform.platform(), 'packages': sorted(["
        "{'name': d.metadata.get('Name', 'unknown'), 'version': d.version} "
        f"for d in m.distributions() if d.metadata.get('Name', '').lower() in {selected!r}], "
        "key=lambda x: x['name'].lower())}))"
    )
    import subprocess

    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect EVA Python environment: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _validate_smoke_target(config, contracts: LoadedContracts, eva_root: Path, eva_python: Path):
    if contracts.policy.name != "mock" or contracts.manifest.source_kind != "mock":
        raise ValueError("the first EVA G2 loopback smoke accepts only XPL's deterministic mock policy")
    if contracts.robot.robot_id != "agibot_g2" or contracts.robot.profile_id != "agibot_g2_eva_fake_v1":
        raise ValueError("the smoke requires the explicitly fake-only agibot_g2_eva_fake_v1 profile")
    if len(contracts.robot.action_fields) != 24 or len(contracts.robot.state_fields) != 24:
        raise ValueError("the fake G2 profile must declare 24 named state and action fields")
    if contracts.robot.control_hz != 30:
        raise ValueError("the EVA G2 Fake source publishes at 30 Hz in this smoke harness")
    ContractValidator.require_profiles(contracts.manifest, contracts.robot)
    fake_node = eva_root / "examples/hardware/agibot_g2/fake_node.py"
    if not fake_node.is_file():
        raise FileNotFoundError(f"EVA G2 Fake node was not found: {fake_node}")
    if not eva_python.is_file():
        raise FileNotFoundError(f"EVA Python interpreter was not found: {eva_python}")
    return fake_node


async def run_fake_smoke(
    *,
    service_config: str | Path,
    eva_root: str | Path,
    eva_python: str | Path,
    frames: int = 30,
    output: str | Path | None = None,
    startup_timeout_s: float = 20.0,
    observation_timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Run the XPL RPC -> EVA Fake G2 loopback without touching device SDKs."""
    if type(frames) is not int or frames < 3:
        raise ValueError("frames must be an integer >= 3")
    if startup_timeout_s <= 0 or observation_timeout_s <= 0:
        raise ValueError("timeouts must be positive")

    project_root = Path(__file__).resolve().parents[3]
    service_path = Path(service_config).expanduser().resolve()
    source_root = Path(eva_root).expanduser().resolve()
    python = Path(eva_python).expanduser().resolve()
    service, contracts = load_service(service_path)
    fake_node = _validate_smoke_target(service, contracts, source_root, python)

    run_id = uuid4().hex
    run_dir = Path(output).expanduser().resolve() if output else project_root / "runs" / "eva-g2-fake" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    service_log_path = run_dir / "policy-server.stderr.log"
    node_log_path = run_dir / "eva-fake-node.log"
    trace_path = run_dir / "trace.jsonl"

    observation_port, action_port = _free_loopback_port(), _free_loopback_port()
    while observation_port == action_port:
        action_port = _free_loopback_port()
    observation_endpoint = f"tcp://127.0.0.1:{observation_port}"
    action_endpoint = f"tcp://127.0.0.1:{action_port}"

    process_env = os.environ.copy()
    python_path = [str(project_root), str(project_root / "src")]
    if process_env.get("PYTHONPATH"):
        python_path.append(process_env["PYTHONPATH"])
    process_env["PYTHONPATH"] = os.pathsep.join(python_path)
    process_env["NO_PROXY"] = process_env["no_proxy"] = "127.0.0.1,localhost"
    eva_env = os.environ.copy()
    eva_env["PYTHONPATH"] = os.pathsep.join([str(source_root), str(source_root / "src")])
    eva_env["NO_PROXY"] = eva_env["no_proxy"] = "127.0.0.1,localhost"
    eva_env["CUDA_VISIBLE_DEVICES"] = ""

    policy_process = None
    fake_process = None
    context = None
    subscriber = publisher = None
    client = None
    controller = None
    events: list[dict[str, Any]] = []
    action_matches_source_state = True
    source_states: dict[int, np.ndarray] = {}
    observations = actions = 0
    timestamps_increasing = True
    previous_timestamp = None
    policy_ready = False
    fake_started = False
    policy_log = service_log_path.open("w", encoding="utf-8")
    fake_log = node_log_path.open("w", encoding="utf-8")
    failure = None

    try:
        policy_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "xpolicy",
            "policy",
            "serve",
            "--config",
            str(service_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            cwd=project_root,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=policy_log,
        )
        ready_line = await asyncio.wait_for(policy_process.stdout.readline(), timeout=startup_timeout_s)
        if not ready_line:
            raise RuntimeError("XPL policy server exited before readiness")
        ready = json.loads(ready_line.decode("utf-8"))
        if ready.get("event") != "policy_server_ready" or not ready.get("url", "").startswith("ws://127.0.0.1:"):
            raise RuntimeError(f"unexpected XPL policy readiness frame: {ready!r}")
        policy_ready = True

        fake_process = await asyncio.create_subprocess_exec(
            str(python),
            str(fake_node),
            "--obs-endpoint",
            observation_endpoint,
            "--action-endpoint",
            action_endpoint,
            "--rate",
            "30",
            "--dynamics-mode",
            "second-order",
            "--dynamics-rate",
            "200",
            "--image-height",
            "64",
            "--image-width",
            "64",
            cwd=source_root,
            env=eva_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=fake_log,
            stderr=asyncio.subprocess.STDOUT,
        )
        fake_started = True

        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.connect(observation_endpoint)
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.LINGER, 0)
        publisher.connect(action_endpoint)
        # PUB/SUB has a slow-joiner window; no startup observation is used as evidence.
        await asyncio.sleep(0.25)

        client = PolicyEvalClient(
            PolicyEvalClientConfig(
                url=ready["url"],
                evaluation_id=f"eva-g2-fake-{run_id}",
                max_connect_attempts=2,
                connect_timeout_s=2,
                handshake_timeout_s=2,
                request_timeout_s=3,
                connect_retry_delay_s=0.2,
                max_connect_seconds=6,
                close_timeout_s=2,
            )
        )
        await client.connect(handshake=True)
        clock_state = {"tick": 0, "timestamp": 0.0, "domain": "eva_host_monotonic"}

        def control_clock():
            return clock_state["tick"], clock_state["timestamp"], clock_state["domain"]

        execution = ExecutionConfig(
            mode="record_only",
            strategy="async_latest_only",
            control_hz=contracts.robot.control_hz,
            execute_horizon=1,
            latency_trim=True,
            overlap_blend_steps=0,
            empty_buffer="skip_record",
        )
        controller = AsyncLatestOnly(
            WebSocketPolicyBackend(client), contracts, execution, control_clock=control_clock
        )
        episode_id = f"{run_id}:0"
        controller.reset(episode_id)
        await controller.wait_idle(timeout_s=5)

        with trace_path.open("x", encoding="utf-8") as trace:
            for frame_id in range(frames):
                if fake_process.returncode is not None:
                    raise RuntimeError(f"EVA Fake G2 exited with code {fake_process.returncode}")
                payload = await asyncio.wait_for(
                    subscriber.recv(), timeout=observation_timeout_s
                )
                wire = decode_eva_observation(payload)
                if previous_timestamp is not None and wire.timestamp_s <= previous_timestamp:
                    timestamps_increasing = False
                    raise ValueError("EVA capture timestamps did not increase")
                previous_timestamp = wire.timestamp_s
                obs = to_xpl_observation(
                    wire, contracts.robot, frame_id=frame_id, episode_id=episode_id
                )
                source_states[frame_id] = obs.state.copy()
                clock_state.update(tick=frame_id, timestamp=obs.timestamp_s)
                controller.refresh_clock()

                scheduled = controller.take(
                    frame_id, now_s=obs.timestamp_s, clock_domain=obs.clock_domain
                )
                if scheduled is not None:
                    prior_state = source_states.get(scheduled.ticket.source_frame_id)
                    if prior_state is None or not np.array_equal(scheduled.values, prior_state):
                        action_matches_source_state = False
                    await publisher.send(encode_fake_action(scheduled.values))
                    actions += 1
                    events.append(
                        {
                            "stage": "fake_action_published",
                            "control_tick": frame_id,
                            "source_frame_id": scheduled.ticket.source_frame_id,
                            "generation": scheduled.ticket.generation,
                            "action_seq": scheduled.action_seq,
                            "target": "isolated_eva_fake_process",
                        }
                    )

                controller.offer(obs, control_tick=frame_id)
                observations += 1
                events.append(
                    {
                        "stage": "observation_received",
                        "frame_id": frame_id,
                        "timestamp_s": obs.timestamp_s,
                        "camera_shapes": {name: list(image.shape) for name, image in obs.images.items()},
                        "state_shape": list(obs.state.shape),
                    }
                )
                trace.write(json.dumps(events[-1], allow_nan=False) + "\n")
                trace.flush()
                await asyncio.sleep(0)

            await controller.wait_idle(timeout_s=5)

        if controller.fault is not None:
            raise RuntimeError(f"XPL async execution fault: {controller.fault}")
        if observations != frames or actions < 1 or controller.predictions < 1:
            raise RuntimeError(
                f"smoke made insufficient progress: observations={observations}, "
                f"predictions={controller.predictions}, actions={actions}"
            )
        if not action_matches_source_state:
            raise RuntimeError("mock hold actions did not preserve the named 24D state layout")

    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if controller is not None:
            controller.stop("smoke_complete")
            try:
                await controller.wait_idle(timeout_s=5)
            except Exception as exc:
                failure = failure or {"type": type(exc).__name__, "message": str(exc)}
            try:
                await controller.close(timeout_s=5)
            except Exception as exc:
                failure = failure or {"type": type(exc).__name__, "message": str(exc)}
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                failure = failure or {"type": type(exc).__name__, "message": str(exc)}
        for process in (fake_process, policy_process):
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        if subscriber is not None:
            subscriber.close(linger=0)
        if publisher is not None:
            publisher.close(linger=0)
        if context is not None:
            context.term()
        policy_log.close()
        fake_log.close()

    eva_files = (
        source_root / "pyproject.toml",
        source_root / "examples/hardware/agibot_g2/node.py",
        source_root / "examples/hardware/fake_common.py",
        source_root / "src/robots/zoo/agibot_g2/__init__.py",
    )
    checks = [
        {"id": "policy_service_ready", "status": "PASS" if policy_ready else "FAIL"},
        {"id": "eva_fake_process_started", "status": "PASS" if fake_started else "FAIL"},
        {"id": "observations_decoded", "status": "PASS" if observations == frames else "FAIL"},
        {"id": "policy_rpc_predictions", "status": "PASS" if controller and controller.predictions > 0 else "FAIL"},
        {"id": "fake_actions_published", "status": "PASS" if actions > 0 else "FAIL"},
        {"id": "named_24d_hold_mapping", "status": "PASS" if action_matches_source_state and actions > 0 else "FAIL"},
        {"id": "source_timestamps_increase", "status": "PASS" if timestamps_increasing else "FAIL"},
        {"id": "real_hardware", "status": "NOT_RUN", "reason": "the harness starts only EVA fake_node.py"},
        {"id": "task_success", "status": "NOT_RUN", "reason": "deterministic mock policy does not evaluate a task"},
    ]
    report = {
        "schema_version": 1,
        "scope": "eva_g2_fake_policy_rpc_smoke",
        "status": "PASS" if failure is None and all(item["status"] in {"PASS", "NOT_RUN"} for item in checks) else "FAIL",
        "run_id": run_id,
        "xpl": {**source_revision(), "service_config_sha256": file_sha256(service_path), "python": sys.version,
                "executable": sys.executable, "packages": _packages()},
        "eva": {"source_revision": None, "revision_note": "local extracted tree has no .git metadata",
                "source_root": str(source_root), "python": _eva_inventory(python),
                "source_file_sha256": {str(path.relative_to(source_root)): file_sha256(path) for path in eva_files}},
        "profile": {"profile_id": contracts.robot.profile_id, "robot_id": contracts.robot.robot_id,
                    "action_type": contracts.robot.action_type.value,
                    "state_dim": len(contracts.robot.state_fields), "action_dim": len(contracts.robot.action_fields),
                    "camera_sources": {camera.name: camera.source_key for camera in contracts.robot.cameras}},
        "results": {"observations": observations,
                    "policy_predictions": controller.predictions if controller is not None else 0,
                    "fake_actions_published": actions,
                    "action_matches_source_observation": action_matches_source_state,
                    "timestamps_increasing": timestamps_increasing},
        "checks": checks,
        "artifacts": {"directory": str(run_dir), "trace": str(trace_path),
                      "policy_server_log": str(service_log_path), "eva_fake_node_log": str(node_log_path)},
        "not_verified": ["real GDK image color semantics", "real G2 gripper joint mapping",
                         "physical units/calibration", "device-local watchdog and stop", "task success"],
        "error": failure,
    }
    write_json(run_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated EVA Fake G2 ↔ XPL policy RPC smoke.")
    parser.add_argument("--service-config", type=Path, default=None)
    parser.add_argument("--eva-root", type=Path, required=True)
    parser.add_argument("--eva-python", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--startup-timeout-s", type=float, default=20.0)
    parser.add_argument("--observation-timeout-s", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    service_config = args.service_config or project_root / "configs/integrations/eva_g2_fake/service.yaml"
    report = asyncio.run(
        run_fake_smoke(
            service_config=service_config,
            eva_root=args.eva_root,
            eva_python=args.eva_python,
            frames=args.frames,
            output=args.output,
            startup_timeout_s=args.startup_timeout_s,
            observation_timeout_s=args.observation_timeout_s,
        )
    )
    print(json.dumps({"status": report["status"], "report": report["artifacts"]["directory"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
