"""Probe an existing simulator interpreter in a supervised child process.

The probes establish package/import/engine bring-up only. They do not construct
RoboDojo/RoboTwin tasks, judge success, modify drivers, or contact robot SDKs.
"""

from __future__ import annotations

import argparse
import faulthandler
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import sys
import time


PACKAGES = ("isaacsim", "isaaclab", "sapien", "torch", "numpy", "websockets", "pydantic", "h5py", "pyyaml", "mplib")
MODES = ("packages", "sapien_cpu", "sapien_render", "isaac_app")
NATIVE_PACKAGES = frozenset({"numpy", "torch", "torchvision", "torchaudio", "triton", "sapien", "mujoco", "jax", "jaxlib"})


def native_constraints(distributions):
    """Snapshot installed native versions, without requesting their installation."""
    versions = {}
    for distribution in distributions:
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in NATIVE_PACKAGES or name.startswith(("nvidia-", "isaacsim", "isaaclab")):
            version = distribution.version
            if name in versions and versions[name] != version:
                raise ValueError(f"conflicting installed versions for protected package {name}")
            if any(character.isspace() for character in version) or any(character in version for character in ";#\\"):
                raise ValueError(f"invalid installed version for protected package {name}")
            versions[name] = version
    return [f"{name}=={version}" for name, version in sorted(versions.items())]


def write_child(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def child(mode, output, gpu):
    # Native SDK failures cannot be caught by Python. Preserve their current
    # Python stack in the supervisor's log instead of losing all context.
    faulthandler.enable()
    report = {"schema_version": 1, "scope": "simulator_runtime_child_probe", "mode": mode, "status": "RUNNING",
              "python": sys.version, "executable": sys.executable, "prefix": sys.prefix, "packages": {},
              "platform": platform.platform(), "libc": list(platform.libc_ver()), "package_wheel_tags": {},
              "stages": [], "module_locations": {},
              "not_verified": ["benchmark_task", "policy_checkpoint", "original_regression", "robot_or_EVA_device", "task_success"]}
    for name in PACKAGES:
        try:
            report["packages"][name] = metadata.version(name)
            wheel = metadata.distribution(name).read_text("WHEEL") or ""
            report["package_wheel_tags"][name] = [line.removeprefix("Tag: ") for line in wheel.splitlines() if line.startswith("Tag: ")]
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    write_child(output, report)
    started = time.monotonic()
    def stage(name, status="PASS", **details):
        report["stages"].append({"name": name, "status": status, **details})
        write_child(output, report)

    try:
        for name in ("isaacsim", "isaaclab", "sapien"):
            spec = util.find_spec(name)
            report["module_locations"][name] = spec.origin if spec else None
        if mode == "packages":
            constraints = native_constraints(metadata.distributions())
            (output.parent / "native-stack-constraints.txt").write_text("\n".join(constraints) + "\n", encoding="utf-8")
            report["native_stack_constraints"] = constraints
            report["stages"].append({"name": "package_metadata", "status": "PASS"})
        elif mode == "sapien_cpu":
            import sapien
            import sapien.physx

            report["stages"].append({"name": "sapien_import", "status": "PASS"})
            write_child(output, report)
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
            scene.set_timestep(1 / 240)
            for _ in range(2):
                scene.step()
            report["stages"].append({"name": "actual_CPU_physics_steps", "status": "PASS", "steps": 2,
                                      "timestep_s": float(scene.get_timestep()), "scene_contents": "empty_engine_probe"})
            report["not_verified"].append("camera_rendering")
        elif mode == "sapien_render":
            stage("sapien_import_started", "RUNNING")
            import numpy as np
            import sapien
            import sapien.physx
            import sapien.render
            stage("sapien_import")

            # Use the genuine RoboTwin camera shader settings. This scene is
            # only a renderer availability probe, not a benchmark surrogate.
            stage("ray_tracing_configuration_started", "RUNNING")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")
            stage("ray_tracing_configuration")
            stage("sapien_render_device_started", "RUNNING", requested_device=f"cuda:{gpu}")
            device = sapien.Device(f"cuda:{gpu}")
            stage("sapien_render_device", device_name=device.name, pci_string=device.pci_string,
                  cuda_id=device.cuda_id, can_render=device.can_render())
            if not device.can_render():
                raise RuntimeError("requested CUDA device is not a SAPIEN rendering device")
            stage("sapien_render_system_started", "RUNNING", requested_device=f"cuda:{gpu}")
            renderer = sapien.render.RenderSystem(device)
            stage("sapien_render_system", requested_device=f"cuda:{gpu}")
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
            stage("sapien_scene")
            material = sapien.render.RenderMaterial(base_color=[.8, .1, .05, 1.])
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[.25, .25, .25], material=material)
            builder.set_initial_pose(sapien.Pose(p=[1., 0., 0.]))
            builder.build_static(name="probe_cube")
            scene.set_ambient_light([.5, .5, .5])
            camera = scene.add_camera("probe", 64, 64, 1., .01, 10.)
            for _ in range(2):
                scene.step()
            scene.update_render()
            camera.take_picture()
            color = np.asarray(camera.get_picture("Color"))
            if color.shape != (64, 64, 4) or not np.isfinite(color).all():
                raise RuntimeError("renderer returned an invalid Color buffer")
            np.savez_compressed(output.parent / "render-color.npz", SDK_Color_RGBA=color)
            report["stages"].append({"name": "actual_ray_traced_camera_frame", "status": "PASS", "shape": list(color.shape),
                                      "dtype": color.dtype.str, "shader": "rt", "samples_per_pixel": 32, "path_depth": 8,
                                      "denoiser": "oidn", "scene_contents": "synthetic_renderer_probe_only"})
        elif mode == "isaac_app":
            # Import after the child report exists: the SDK may call sys.exit
            # or crash natively during its own bootstrap.
            stage("isaac_import_started", "RUNNING")
            from isaacsim import SimulationApp

            report["stages"].append({"name": "isaac_import", "status": "PASS"})
            write_child(output, report)
            app = None
            try:
                app = SimulationApp({"headless": True, "width": 64, "height": 64,
                                     "active_gpu": gpu, "physics_gpu": gpu, "multi_gpu": False,
                                     "sync_loads": True})
                for _ in range(2):
                    app.update()
                if not app.is_running():
                    raise RuntimeError("Isaac application is not running after startup updates")
                report["stages"].append({"name": "actual_isaac_application_boot", "status": "PASS", "updates": 2,
                                          "headless": True, "requested_gpu_index": gpu})
                write_child(output, report)
            finally:
                if app is not None:
                    app.close()
                    report["stages"].append({"name": "isaac_application_close", "status": "PASS"})
            report["not_verified"].append("task_camera_rendering")
        else:
            raise ValueError("unknown simulator probe mode")
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "BLOCKED" if isinstance(exc, (ImportError, SystemExit)) else "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["duration_s"] = time.monotonic() - started
        write_child(output, report)
    return 0 if report["status"] == "PASS" else 2


def run_child(command, cwd, environment, timeout_s, log_path):
    """Kill only this owned process group on timeout; retain native exits/logs."""
    # Lazy import keeps the frozen child usable in an SDK interpreter without
    # installing the public core or its incompatible WebSocket dependency.
    from xpolicy.core.process import run_owned_process
    return run_owned_process(command, cwd, environment, timeout_s, log_path)


def probe(python, mode, output, *, timeout_s=120., gpu=0):
    from xpolicy.core.artifacts import file_sha256, source_revision, write_json
    from xpolicy.core import process as process_module

    if mode not in MODES or not math.isfinite(timeout_s) or not 0 < timeout_s <= 3600 or isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("a supported mode, finite timeout in (0, 3600], and nonnegative integer gpu index are required")
    python = python.expanduser().absolute()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    child_output = output / "child.json"
    child_script = output / "probe_child.py"
    shutil.copyfile(Path(__file__).resolve(), child_script)
    command = [str(python), "-B", str(child_script), "--child", mode,
               "--child-output", str(child_output), "--gpu", str(gpu)]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    # Kit/SAPIEN logs and caches stay under the new run directory where the
    # SDK supports the ordinary XDG cache location. SDK-specific caches/logs
    # may still use their documented defaults, which the captured log exposes.
    environment["XDG_CACHE_HOME"] = str(output / "cache")
    report = {"schema_version": 1, "scope": "simulator_runtime_bringup", "mode": mode, "status": "RUNNING",
              "provenance": {**source_revision(), "probe_source_sha256": file_sha256(Path(__file__)),
                             "supervisor_source_sha256": file_sha256(Path(process_module.__file__))},
              "gpu_index_requested": gpu,
              "inherited_runtime_flags": {name: environment.get(name) for name in ("CUDA_VISIBLE_DEVICES", "OMNI_KIT_ACCEPT_EULA")},
              "not_verified": ["full_Runner_simulation_provider", "benchmark_task", "camera_contract", "trained_policy",
                               "original_benchmark_regression", "EVA_or_robot_hardware", "task_success"]}
    write_json(output / "SIMULATOR_PROBE_REPORT.json", report)
    try:
        if not python.is_file():
            report["status"] = "BLOCKED"
            report["error"] = {"type": "InterpreterMissing", "message": "simulator interpreter is not materialized"}
            return report
        report["provenance"]["interpreter_sha256"] = file_sha256(python.resolve())
        execution = run_child(command, output, environment, timeout_s, output / "runtime.log")
        report["execution"] = execution
        evidence = json.loads(child_output.read_text()) if child_output.is_file() else None
        report["child"] = evidence
        if evidence is not None and (not isinstance(evidence, dict) or evidence.get("scope") != "simulator_runtime_child_probe"
                                     or evidence.get("mode") != mode or evidence.get("status") not in {"RUNNING", "PASS", "FAIL", "BLOCKED"}):
            raise ValueError("child evidence has an invalid scope, mode or status")
        if execution["timed_out"]:
            report["status"] = "BLOCKED"
            report["error"] = {"type": "TimeoutError", "message": "owned simulator process exceeded its bring-up budget"}
        elif execution["exit_code"] != 0:
            report["status"] = evidence["status"] if evidence and evidence["status"] in {"BLOCKED", "FAIL"} else "FAIL"
            code = execution["exit_code"]
            report["error"] = {"type": "NativeProcessExit" if code is not None and code < 0 else "ChildProcessExit", "exit_code": code,
                               "signal": signal.Signals(-code).name if code is not None and code < 0 else None,
                               "child_error": evidence.get("error") if evidence else None}
        elif evidence is None or evidence.get("status") != "PASS":
            report["status"] = "FAIL"
            report["error"] = {"type": "InvalidProbeEvidence", "message": "successful process exit lacks successful child evidence"}
        else:
            report["status"] = "PASS"
    except (OSError, ValueError, RuntimeError) as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except KeyboardInterrupt:
        report["status"] = "BLOCKED"
        report["error"] = {"type": "SupervisorInterrupted", "message": "probe interrupted; owned process termination attempted"}
    finally:
        report["source_stable"] = (report["provenance"]["probe_source_sha256"] == file_sha256(Path(__file__)) and
                                   report["provenance"]["supervisor_source_sha256"] == file_sha256(Path(process_module.__file__)))
        if not report["source_stable"]:
            report["status"] = "FAIL"
        report["evidence_sha256"] = {path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "SIMULATOR_PROBE_REPORT.json"}
        write_json(output / "SIMULATOR_PROBE_REPORT.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--mode", choices=MODES, default="packages")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-s", type=float, default=120.)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child is not None:
        return child(args.child, args.child_output, args.gpu)
    if args.python is None or args.output is None:
        parser.error("--python and --output are required")
    if not 0 < args.timeout_s <= 3600 or args.gpu < 0:
        parser.error("timeout must be finite in (0, 3600] and gpu index nonnegative")
    report = probe(args.python, args.mode, args.output, timeout_s=args.timeout_s, gpu=args.gpu)
    print(json.dumps({"report": str(args.output / "SIMULATOR_PROBE_REPORT.json"), "status": report["status"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
