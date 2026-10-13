"""Noncausal dataset replay through the shared record-only execution owner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from xpolicy.core.artifacts import file_sha256, source_revision, verify_artifact, write_json
from xpolicy.core.config import load_experiment
from xpolicy.core.contracts import ContractValidator
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.dataset.hdf5 import Hdf5Dataset
from xpolicy.execution.async_latest import AsyncLatestOnly, LocalPolicyBackend
from xpolicy.execution.buffer import ActionBuffer
from xpolicy.execution.record_only import RecordOnlySink
from xpolicy.policy.runtime import build_policy


def _preflight(config, contracts):
    if config.environment.kind != "dataset_direct":
        raise UnsupportedCapability("offline runner currently supports dataset_direct only")
    if config.execution.mode != "record_only":
        raise UnsupportedCapability("dataset replay requires record_only execution")
    if config.policy.api != "chunk":
        raise UnsupportedCapability("dataset replay requires an explicit complete-chunk policy")
    if config.environment.episode != 0:
        raise ValueError("xpl_hdf5_v1 selects one trajectory file; episode must be 0")
    ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
    if contracts.dataset is None:
        raise ValueError("dataset_direct requires a dataset profile")
    ActionBuffer(contracts.manifest, config.execution)
    if config.execution.control_hz != contracts.robot.control_hz or config.execution.control_hz != contracts.dataset.sample_hz:
        raise ValueError("robot control rate and dataset sample rate must agree; resampling is not implemented")
    dataset_path = verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
    if config.environment.dataset_path != dataset_path:
        raise ValueError("dataset_path must match the verified dataset artifact")
    return dataset_path


class _Trial:
    def __init__(self, run_id, repeat, config, contracts, source, directory, trace):
        self.run_id, self.trial_id = run_id, f"{run_id}:{repeat}"
        self.config, self.contracts, self.source, self.directory, self.trace = config, contracts, source, directory, trace
        self.repeat, self.generation, self.row = repeat, repeat + 1, 0
        self.started = time.monotonic()
        self.sink = RecordOnlySink()
        self.predictions, self.recorded, self.observations, self.empty_ticks, self.skipped_frames = 0, 0, 0, 0, 0
        self.prediction_requests = 0
        self.superseded_observations, self.errors, self.fault = 0, [], None
        self.secondary_faults = []
        self.cleanup_drained = True

    def check_budget(self):
        if time.monotonic() - self.started >= self.config.experiment.max_duration_s:
            raise TimeoutError("trial exceeded max_duration_s; no further action may be recorded")

    def event(self, stage, obs=None, **payload):
        identity = {"run_id": self.run_id, "trial_id": self.trial_id, "episode_id": self.trial_id,
                    "episode_generation": self.generation, "monotonic_s": time.monotonic(), "stage": stage,
                    "source_row": self.row, "frame_id": obs.frame_id if obs is not None else None,
                    "profile_id": self.contracts.robot.profile_id}
        self.trace.write(json.dumps({**identity, **payload}, allow_nan=False) + "\n")
        self.trace.flush()

    def execution_event(self, value):
        value = dict(value)
        stage = value.pop("stage")
        if stage == "prediction":
            self.predictions += 1
        elif stage == "inference_request":
            self.prediction_requests += 1
        elif stage == "execution_empty":
            self.empty_ticks += 1
        if not self.config.artifacts.save_actions and "values" in value:
            value["values"] = None
        self.event(stage, **value)

    def observe(self, row):
        self.row = row
        obs = self.source.observe(row, self.trial_id)
        ContractValidator.observation(obs, self.contracts.robot, self.contracts.manifest)
        self.observations += 1
        self.event("observation", obs, timestamp_s=obs.timestamp_s, clock_domain=obs.clock_domain,
                   state=obs.state.tolist(), state_names=obs.state_names, state_units=obs.state_units,
                   images={name: {"sha256": hashlib.sha256(image.tobytes()).hexdigest(), "shape": image.shape, "color_space": "RGB"} for name, image in obs.images.items()})
        if self.config.artifacts.save_images and self.config.validation.trace_level == "full":
            np.savez_compressed(self.directory / f"input-{self.repeat}-{row}.npz", state=obs.state,
                                **{f"image_{index}": image for index, image in enumerate(obs.images.values())})
        return obs

    def record(self, commands, observation, control_row):
        if not commands:
            return
        self.check_budget()
        labels = self.source.labels(control_row, len(commands))
        accepted = self.sink.submit_commands(commands)
        self.recorded += len(commands)
        self.event("record_only_submission", observation, **commands[0].ticket.context(),
                   control_frame_id=int(self.source.frame_ids[control_row]), control_tick=control_row,
                   action_seq=commands[0].action_seq, values=accepted.tolist() if self.config.artifacts.save_actions else None,
                   count=len(commands), status="recorded", causal=False, device_feedback=None)
        if labels is not None:
            # Wider arithmetic plus scaled reductions avoids float64 square/
            # sum overflow. Unrepresentable metrics are reported explicitly.
            with np.errstate(over="ignore", invalid="ignore"):
                self.errors.append(accepted.astype(np.longdouble) - labels.astype(np.longdouble))

    def fail(self, exc):
        fault = {"type": type(exc).__name__, "message": str(exc)}
        if self.fault is None:
            self.fault = fault
            self.event("fault", fault=fault)
        else:
            self.secondary_faults.append(fault)
            self.event("secondary_fault", fault=fault)

    def finish(self):
        result = {"trial_id": self.trial_id, "status": "FAULT" if self.fault else "COMPLETED", "fault": self.fault,
                  "dataset_frames": self.source.length, "recorded_action_steps": self.recorded,
                  "observations": self.observations, "predictions": self.predictions,
                  "prediction_requests": self.prediction_requests,
                  "empty_buffer_ticks": self.empty_ticks, "skipped_recorded_frames": self.skipped_frames,
                  "superseded_observations": self.superseded_observations, "duration_s": time.monotonic() - self.started,
                  "execution_strategy": self.config.execution.strategy, "inference_cleanup_drained": self.cleanup_drained,
                  "secondary_faults": self.secondary_faults,
                  "task_success": None, "success_source": "not_measured", "causal": False}
        if self.errors:
            error = np.concatenate(self.errors, axis=0)
            metrics = {}
            for index, field in enumerate(self.contracts.manifest.action_fields):
                values = np.abs(error[:, index])
                scale = np.max(values)
                with np.errstate(over="ignore", invalid="ignore"):
                    mae = float(scale * np.mean(values / scale)) if scale else 0.0
                    rmse = float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0
                valid = np.isfinite(mae) and np.isfinite(rmse)
                metrics[field.name] = {"unit": field.unit, "mae": mae if valid else None, "rmse": rmse if valid else None,
                                       "samples": len(error), "metric_status": "COMPUTED" if valid else "EXCEEDS_FLOAT_RANGE"}
            result["behavior_cloning_error"] = metrics
        self.event("trial_end", result=result)
        return result


def _sync_trial(trial, model, buffer):
    buffer.emit = trial.execution_event
    trial.generation = buffer.reset(trial.trial_id)
    try:
        model.reset()
        row = 0
        while row < trial.source.length:
            trial.check_budget()
            obs = trial.observe(row)
            request = buffer.issue(obs, control_tick=row)
            started = time.monotonic()
            chunk = model.predict_action_chunk(obs)
            trial.check_budget()
            trial.execution_event({"stage": "prediction", **request.context(), "trace_layer": "A0",
                                   "values": chunk.values.tolist(), "shape": chunk.values.shape,
                                   "action_type": chunk.action_type.value, "action_units": chunk.action_units,
                                   "joint_names": chunk.joint_names, "latency_s": time.monotonic() - started})
            buffer.accept(request, chunk, control_tick=row, now_s=obs.timestamp_s, clock_domain=obs.clock_domain)
            count = min(trial.config.execution.execute_horizon, trial.source.length - row)
            commands = []
            for control_row in range(row, row + count):
                command = buffer.take(control_row, now_s=float(trial.source.timestamps[control_row]), clock_domain=trial.source.profile.clock_domain)
                if command is not None:
                    commands.append(command)
                else:
                    raise RuntimeError("sync prediction did not provide an executable record-only step")
            trial.record(commands, obs, row)
            row += count
        verify_artifact(trial.contracts.dataset.artifact, trial.contracts.dataset_profile_path)
    except Exception as exc:
        buffer.clear("fault")
        trial.fail(exc)
    return trial.finish()


async def _async_trials(trials, backend, config, contracts):
    controller = AsyncLatestOnly(backend, contracts, config.execution)
    results = []
    try:
        for trial in trials:
            trial.started = time.monotonic()
            controller.emit = trial.execution_event
            trial.generation = controller.reset(trial.trial_id)
            try:
                await controller.wait_idle(max(0, config.experiment.max_duration_s - (time.monotonic() - trial.started)))
                if controller.fault is not None:
                    raise RuntimeError(controller.fault["message"])
                control_started = time.monotonic()

                def clock():
                    trial.check_budget()
                    current_tick = int((time.monotonic() - control_started) * config.execution.control_hz)
                    now_s = float(trial.source.timestamps[current_tick]) if current_tick < trial.source.length else None
                    return current_tick, now_s, trial.source.profile.clock_domain

                controller.control_clock = clock
                row = 0
                while row < trial.source.length:
                    trial.check_budget()
                    due = control_started + row / config.execution.control_hz
                    await asyncio.sleep(max(0, due - time.monotonic()))
                    if controller.fault is not None:
                        trial.fault = controller.fault
                        break
                    trial.check_budget()
                    actual_tick = max(row, clock()[0], controller.buffer.current_tick)
                    if actual_tick >= trial.source.length:
                        trial.skipped_frames += trial.source.length - row
                        break
                    if actual_tick > row:
                        trial.skipped_frames += actual_tick - row
                        trial.event("replay_tick_gap", skipped_rows=list(range(row, actual_tick)), control_tick=actual_tick)
                        row = actual_tick
                    obs = trial.observe(row)
                    controller.offer(obs, control_tick=row)
                    # Let the worker start or complete without blocking sensor
                    # replay on inference. Control ticks still come from clock.
                    await asyncio.sleep(0)
                    if controller.fault is not None:
                        trial.fault = controller.fault
                        break
                    actual_tick = max(row, clock()[0], controller.buffer.current_tick)
                    if actual_tick >= trial.source.length:
                        trial.skipped_frames += trial.source.length - row
                        break
                    if actual_tick > row:
                        trial.skipped_frames += actual_tick - row
                        trial.event("observation_decode_late", observation_row=row, control_tick=actual_tick)
                    command = controller.take(actual_tick, now_s=float(trial.source.timestamps[actual_tick]), clock_domain=trial.source.profile.clock_domain)
                    if command is not None:
                        trial.record([command], obs, actual_tick)
                    row = actual_tick + 1
                controller.stop("trial_end")
                remaining = max(0, config.experiment.max_duration_s - (time.monotonic() - trial.started))
                await controller.wait_idle(remaining)
                if controller.fault is not None:
                    trial.fault = controller.fault
                verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
            except Exception as exc:
                controller.stop("trial_fault")
                if controller.fault is not None:
                    trial.fault = controller.fault
                trial.fail(exc)
            trial.superseded_observations = controller.superseded_observations
            if controller._worker is not None and not controller._worker.done():
                trial.cleanup_drained = False
                # Do not begin another trial while its model call is live.
                results.append(trial.finish())
                break
            results.append(trial.finish())
    finally:
        await controller.close(timeout_s=0)
    return results


def run_experiment(path: str | Path) -> dict:
    config, contracts = load_experiment(path)
    dataset_path = _preflight(config, contracts)
    source = Hdf5Dataset(dataset_path, contracts.dataset)
    model, backend, report, cleanup_errors = None, None, None, []
    random_state = np.random.get_state()
    directory = config.artifacts.root / uuid4().hex
    try:
        directory.mkdir(parents=True, exist_ok=False)
        run_id = directory.name
        provenance = {**source_revision(), "configuration_sha256": file_sha256(contracts.source_path),
                      "manifest_sha256": file_sha256(contracts.manifest_path), "robot_profile_sha256": file_sha256(contracts.robot_path),
                      "dataset_profile_sha256": file_sha256(contracts.dataset_profile_path), "dataset_sha256": contracts.dataset.artifact.sha256,
                      "checkpoint_sha256": contracts.manifest.checkpoint.sha256 if contracts.manifest.checkpoint else None, "seed": config.experiment.seed}
        package_root = Path(__file__).resolve().parents[1]
        provenance["core_source_sha256"] = {item.relative_to(package_root).as_posix(): file_sha256(item) for item in sorted(package_root.rglob("*.py"))}
        write_json(directory / "config.json", config.model_dump(mode="json"))
        write_json(directory / "profiles.json", {"model": contracts.manifest.model_dump(mode="json"), "robot": contracts.robot.model_dump(mode="json"), "dataset": contracts.dataset.model_dump(mode="json")})
        write_json(directory / "software-inventory.json", {"packages": sorted([{"name": item.metadata.get("Name", "unknown"), "version": item.version} for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower())})
        np.random.seed(config.experiment.seed)
        with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
            trials = [_Trial(run_id, repeat, config, contracts, source, directory, trace) for repeat in range(config.experiment.repeats)]
            try:
                model = build_policy(contracts)
            except Exception as exc:
                results = []
                for trial in trials:
                    trial.fail(exc)
                    results.append(trial.finish())
            else:
                # Start each budget when its trial begins, not when the batch
                # of repeat objects is allocated or the model is constructed.
                if config.execution.strategy == "sync":
                    buffer = ActionBuffer(contracts.manifest, config.execution)
                    results = []
                    for trial in trials:
                        trial.started = time.monotonic()
                        results.append(_sync_trial(trial, model, buffer))
                else:
                    backend = LocalPolicyBackend(model)
                    results = asyncio.run(_async_trials(trials, backend, config, contracts))
        report = {"schema_version": 1, "scope": "dataset_direct_record_only", "run_id": run_id,
                  "status": "FAULT" if any(item["status"] == "FAULT" for item in results) else "COMPLETED",
                  "artifacts": str(directory), "dataset_source_kind": contracts.dataset.source_kind,
                  "timebase_source": source.timebase_source, "execution_strategy": config.execution.strategy,
                  "control_timebase": "logical_control_ticks" if config.execution.strategy == "sync" else "paced_local_monotonic_control_ticks",
                  "provenance": provenance, "trials": results, "requested_repeats": config.experiment.repeats,
                  "not_run": ["genuine_training_loader_parity", "dataset_wire", "actuator_feedback", "task_success", "hardware_safety"]}
    finally:
        try:
            source.close()
        except Exception as exc:
            cleanup_errors.append({"resource": "dataset", "type": type(exc).__name__, "message": str(exc)})
        if model is not None:
            try:
                if backend is not None:
                    backend.close().result(timeout=1)
                else:
                    model.close()
            except Exception as exc:
                cleanup_errors.append({"resource": "policy", "type": type(exc).__name__, "message": str(exc)})
        np.random.set_state(random_state)
    if report is not None:
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "FAULT"
        write_json(directory / "results.json", report)
    return report
