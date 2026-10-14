"""Noncausal dataset replay through the shared record-only execution owner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import time
import threading
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from xpolicy.core.artifacts import file_sha256, source_revision, verify_artifact, write_json
from xpolicy.core.config import load_experiment
from xpolicy.core.contracts import ContractValidator
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.tasks import resolve_task, validate_task
from xpolicy.dataset.hdf5 import Hdf5Dataset
from xpolicy.execution.async_latest import AsyncLatestOnly, LocalPolicyBackend
from xpolicy.execution.buffer import ActionBuffer
from xpolicy.execution.record_only import RecordOnlySink
from xpolicy.policy.runtime import build_policy
from xpolicy.evaluation.lifecycle import StopRequested, StopSignal, TrialLifecycle, TrialState
from xpolicy.evaluation.metrics import process_cpu_seconds, series


_EVALUATION_LOCK = threading.Lock()


class _EvaluationLease:
    def __init__(self):
        self.acquired = _EVALUATION_LOCK.acquire(blocking=False)
        self.deferred = False

    def defer_until_closed(self, future, random_state):
        self.deferred = True

        def release(_):
            try:
                np.random.set_state(random_state)
            finally:
                _EVALUATION_LOCK.release()

        future.add_done_callback(release)


def _preflight(config, contracts):
    validate_task(config, contracts.manifest)
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
    def __init__(self, run_id, repeat, config, contracts, source, directory, trace, *, stop_signal=None):
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
        self.task = resolve_task(config, contracts.manifest)
        self.stop_signal = stop_signal or StopSignal()
        self.timeout_s = min(config.experiment.max_duration_s, self.task.timeout_s or config.experiment.max_duration_s)
        self.inference_durations, self.queue_delays, self.observation_ages, self.stop_latencies = [], [], [], []
        self.received_at = {}
        self.cpu_started = None
        self.hooks = {"prepare_case": "NOT_RUN", "on_trial_end": "NOT_RUN"}
        self._result = None
        self.lifecycle = TrialLifecycle(self.event)

    def begin(self):
        self.started = time.monotonic()
        self.cpu_started = process_cpu_seconds()

    def case_metadata(self):
        return {"run_id": self.run_id, "trial_id": self.trial_id, "episode_id": self.trial_id,
                "repeat_index": self.repeat, "task": self.task.model_dump(mode="json"),
                "seed": self.config.experiment.seed, "mode": "dataset_direct_record_only"}

    def hook_outcome(self):
        return {**self.case_metadata(), "stop_reason": self.lifecycle.stop_reason, "fault": self.fault,
                "recorded_action_steps": self.recorded, "predictions": self.predictions,
                "task_success": None, "success_source": "not_measured"}

    def complete_hook(self, name):
        self.hooks[name] = "COMPLETED"
        self.event("policy_hook", hook=name, status="COMPLETED", physical_reset=None)

    def check_budget(self):
        request = self.stop_signal.request_info
        if request is not None:
            raise StopRequested(request["reason"])
        if time.monotonic() - self.started >= self.timeout_s:
            raise TimeoutError("trial exceeded its effective task/experiment timeout; no further action may be recorded")

    def ready(self):
        self.check_budget()
        self.lifecycle.transition(TrialState.READY)
        self.event("offline_readiness", dataset_cursor="episode_start", policy_context_reset=True,
                   physical_scene_reset=None, physical_robot_reset=None, scope="dataset_direct_record_only")
        self.lifecycle.transition(TrialState.RUNNING)

    def stop(self, reason):
        if self.lifecycle.stop(reason):
            request = self.stop_signal.request_info
            if request is not None:
                self.stop_latencies.append(max(0., time.monotonic() - request["monotonic_s"]))

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
            self.queue_delays.append(float(value.get("queue_delay_ticks", 0)))
            received = self.received_at.get(value.get("frame_id"))
            if received is not None:
                age = max(0., time.monotonic() - received)
                self.observation_ages.append(age)
                value["local_observation_age_s"] = age
                value["observation_receipt_monotonic_s"] = received
        elif stage == "inference_timing":
            self.inference_durations.append(value["duration_s"])
        elif stage == "execution_empty":
            self.empty_ticks += 1
        if not self.config.artifacts.save_actions and "values" in value:
            value["values"] = None
        self.event(stage, **value)

    def observe(self, row):
        self.row = row
        obs = self.source.observe(row, self.trial_id)
        source_instruction = obs.instruction
        if self.task.instruction.source == "override":
            obs = replace(obs, instruction=self.task.instruction.text)
        self.received_at[obs.frame_id] = time.monotonic()
        ContractValidator.observation(obs, self.contracts.robot, self.contracts.manifest)
        self.observations += 1
        self.event("observation", obs, timestamp_s=obs.timestamp_s, clock_domain=obs.clock_domain,
                   local_receipt_monotonic_s=self.received_at[obs.frame_id],
                   instruction=obs.instruction, source_instruction=source_instruction,
                   instruction_source=self.task.instruction.source,
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
        for name, status in self.hooks.items():
            if status == "RUNNING":
                self.hooks[name] = "FAIL"
        if isinstance(exc, StopRequested):
            self.stop(exc.reason if self.fault is None else "fault")
            return
        fault = {"type": type(exc).__name__, "message": str(exc)}
        self.capture_fault(fault)

    def capture_fault(self, fault):
        if fault["type"] == "StopRequested":
            self.stop(fault["message"] if self.fault is None else "fault")
            return
        if self.fault is None:
            self.fault = dict(fault)
            self.event("fault", fault=fault)
            self.stop("timeout" if fault["type"] == "TimeoutError" else "fault")
        else:
            self.secondary_faults.append(dict(fault))
            self.event("secondary_fault", fault=fault)

    def block(self, exc):
        self.fault = {"type": type(exc).__name__, "message": str(exc)}
        self.event("preflight_failure", fault=self.fault)
        self.lifecycle.transition(TrialState.BLOCKED)

    def finish(self):
        if self._result is not None:
            return self._result
        available = {"observations": self.observations > 0, "predictions": self.predictions > 0,
                     "commands": self.recorded > 0,
                     "images": self.observations > 0 and len(list(self.directory.glob(f"input-{self.repeat}-*.npz"))) == self.observations,
                     "feedback": False}
        missing = [name for name in self.task.required_evidence if not available[name]]
        if missing and self.fault is None and self.lifecycle.stop_reason in {None, "dataset_end"} and self.lifecycle.state != TrialState.BLOCKED:
            self.fail(ValueError(f"required trial evidence was not captured: {missing}"))
        if self.lifecycle.state != TrialState.BLOCKED:
            self.stop("fault" if self.fault else "dataset_end")
            self.lifecycle.finalize(failed=self.fault is not None)
        status = "BLOCKED" if self.lifecycle.state == TrialState.BLOCKED else "FAULT" if self.fault else "STOPPED" if self.lifecycle.stop_reason in {"operator_stop", "interrupted"} else "COMPLETED"
        result = {"trial_id": self.trial_id, "status": status, "fault": self.fault,
                  "lifecycle": self.lifecycle.report(), "task_id": self.task.task_id,
                  "effective_timeout_s": self.timeout_s,
                  "policy_hooks": dict(self.hooks),
                  "missing_required_evidence": missing,
                  "dataset_frames": self.source.length if self.source is not None else None, "recorded_action_steps": self.recorded,
                  "observations": self.observations, "predictions": self.predictions,
                  "prediction_requests": self.prediction_requests,
                  "empty_buffer_ticks": self.empty_ticks, "skipped_recorded_frames": self.skipped_frames,
                  "superseded_observations": self.superseded_observations, "duration_s": time.monotonic() - self.started,
                  "execution_strategy": self.config.execution.strategy, "inference_cleanup_drained": self.cleanup_drained,
                  "secondary_faults": self.secondary_faults,
                  "task_success": None, "success_source": "not_measured", "causal": False}
        result["metrics"] = {
            "inference_duration": series(self.inference_durations),
            "queue_delay": series(self.queue_delays, unit="control_ticks", scope="execution_control_ticks"),
            "local_observation_age": series(self.observation_ages, scope="local_receipt_to_model_dispatch"),
            "local_stop_latency": series(self.stop_latencies, scope="stop_request_to_host_scheduler_stop"),
            **{name: series([], scope="not_available_in_record_only") for name in
               ("sensor_observation_age", "command_to_feedback", "device_halt_latency")},
            "feedback_error": series([], unit="per_field", scope="actuator_feedback_not_measured"),
            "gpu_occupancy": series([], unit="fraction", scope="not_measured"),
        }
        cpu_end = process_cpu_seconds()
        cpu_values = []
        if self.cpu_started is not None and cpu_end is not None and result["duration_s"] > 0:
            cpu_values = [max(0., cpu_end - self.cpu_started) / result["duration_s"]]
        result["metrics"]["cpu_occupancy"] = series(cpu_values, unit="cpu_core_equivalents", scope="runner_process_user_plus_system_time_over_wall_time")
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
        self._result = result
        return result


def _sync_trial(trial, model, buffer):
    buffer.emit = trial.execution_event
    trial.generation = buffer.reset(trial.trial_id)
    try:
        trial.check_budget()
        model.reset_episode(trial.trial_id)
        trial.hooks["prepare_case"] = "RUNNING"
        model.prepare_case(trial.case_metadata())
        trial.complete_hook("prepare_case")
        trial.ready()
        row = 0
        while row < trial.source.length:
            trial.check_budget()
            obs = trial.observe(row)
            request = buffer.issue(obs, control_tick=row)
            started = time.monotonic()
            outcome = "error"
            try:
                chunk = model.predict_action_chunk(obs)
                outcome = "returned"
            finally:
                trial.execution_event({"stage": "inference_timing", **request.context(),
                                       "duration_s": time.monotonic() - started, "outcome": outcome})
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
        buffer.clear("dataset_end")
        trial.stop("dataset_end")
    except StopRequested as exc:
        buffer.clear(exc.reason)
        trial.fail(exc)
    except KeyboardInterrupt:
        trial.stop_signal.request("interrupted")
        buffer.clear("interrupted")
        trial.fail(StopRequested("interrupted"))
    except Exception as exc:
        buffer.clear("fault")
        trial.fail(exc)
    try:
        trial.hooks["on_trial_end"] = "RUNNING"
        model.on_trial_end(trial.hook_outcome())
        trial.complete_hook("on_trial_end")
    except Exception as exc:
        trial.hooks["on_trial_end"] = "FAIL"
        trial.fail(exc)
    return trial.finish()


async def _async_trials(trials, backend, config, contracts):
    controller = AsyncLatestOnly(backend, contracts, config.execution)
    results = []
    try:
        for trial in trials:
            trial.begin()
            controller.emit = trial.execution_event
            trial.generation = controller.reset(trial.trial_id)
            try:
                trial.check_budget()
                remaining = max(0, trial.timeout_s - (time.monotonic() - trial.started))
                idle = asyncio.create_task(controller.wait_idle(remaining))
                stopping = asyncio.create_task(trial.stop_signal.wait_async(remaining))
                try:
                    await asyncio.wait((idle, stopping), return_when=asyncio.FIRST_COMPLETED)
                    request = trial.stop_signal.request_info
                    if request is not None:
                        controller.stop(request["reason"])
                        raise StopRequested(request["reason"])
                    await idle
                finally:
                    stopping.cancel()
                    # wait_idle shields the actual model worker. Cancel this
                    # waiter without cancelling vendor reset/inference.
                    if not idle.done():
                        idle.cancel()
                    await asyncio.gather(idle, stopping, return_exceptions=True)
                if controller.fault is not None:
                    trial.capture_fault(controller.fault)
                    raise RuntimeError(controller.fault["message"])
                trial.hooks["prepare_case"] = "RUNNING"
                await asyncio.wait_for(backend.prepare_case(trial.case_metadata()), timeout=max(0, trial.timeout_s - (time.monotonic() - trial.started)))
                trial.complete_hook("prepare_case")
                trial.ready()
                control_started = time.monotonic()

                def clock():
                    request = trial.stop_signal.request_info
                    if request is not None:
                        controller.stop(request["reason"])
                    else:
                        trial.check_budget()
                    current_tick = int((time.monotonic() - control_started) * config.execution.control_hz)
                    now_s = float(trial.source.timestamps[current_tick]) if current_tick < trial.source.length else None
                    return current_tick, now_s, trial.source.profile.clock_domain

                controller.control_clock = clock
                row = 0
                while row < trial.source.length:
                    trial.check_budget()
                    due = control_started + row / config.execution.control_hz
                    delay = max(0, due - time.monotonic())
                    if delay:
                        remaining = max(0, trial.timeout_s - (time.monotonic() - trial.started))
                        await trial.stop_signal.wait_async(min(delay, remaining))
                    if controller.fault is not None:
                        trial.capture_fault(controller.fault)
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
                        trial.capture_fault(controller.fault)
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
                trial.stop("fault" if trial.fault else "dataset_end")
                remaining = max(0, trial.timeout_s - (time.monotonic() - trial.started))
                await controller.wait_idle(remaining)
                if controller.fault is not None:
                    trial.capture_fault(controller.fault)
                verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
            except StopRequested as exc:
                controller.stop(exc.reason)
                if controller.fault is not None:
                    trial.capture_fault(controller.fault)
                trial.fail(exc)
                try:
                    await controller.wait_idle(max(0, trial.timeout_s - (time.monotonic() - trial.started)))
                except Exception as error:
                    trial.fail(error)
            except (KeyboardInterrupt, asyncio.CancelledError):
                trial.stop_signal.request("interrupted")
                controller.stop("interrupted")
                trial.fail(StopRequested("interrupted"))
                try:
                    await controller.wait_idle(max(0, trial.timeout_s - (time.monotonic() - trial.started)))
                except Exception as exc:
                    trial.fail(exc)
            except Exception as exc:
                controller.stop("trial_fault")
                if controller.fault is not None:
                    trial.capture_fault(controller.fault)
                trial.fail(exc)
            trial.superseded_observations = controller.superseded_observations
            if (controller._worker is not None and not controller._worker.done()) or backend.hook_pending:
                trial.cleanup_drained = False
                # Do not begin another trial while its model call is live.
                results.append(trial.finish())
                break
            try:
                remaining = max(0, trial.timeout_s - (time.monotonic() - trial.started))
                if remaining:
                    trial.hooks["on_trial_end"] = "RUNNING"
                    await asyncio.wait_for(backend.on_trial_end(trial.hook_outcome()), timeout=remaining)
                    trial.complete_hook("on_trial_end")
            except Exception as exc:
                trial.hooks["on_trial_end"] = "FAIL"
                trial.fail(exc)
            if backend.hook_pending:
                trial.cleanup_drained = False
                results.append(trial.finish())
                break
            results.append(trial.finish())
            if trial.stop_signal.request_info is not None:
                break
    finally:
        await controller.close(timeout_s=0)
    return results


def _run_experiment(path: str | Path, *, stop_signal=None, lease=None, admission_error=None) -> dict:
    config, contracts = load_experiment(path)
    stop_signal = stop_signal or StopSignal()
    source, model, backend, report, cleanup_errors, preflight_error = None, None, None, None, [], None
    random_state = np.random.get_state() if admission_error is None else None
    directory = config.artifacts.root / uuid4().hex
    try:
        directory.mkdir(parents=True, exist_ok=False)
        run_id = directory.name
        provenance = {**source_revision(), "configuration_sha256": file_sha256(contracts.source_path),
                      "manifest_sha256": file_sha256(contracts.manifest_path), "robot_profile_sha256": file_sha256(contracts.robot_path),
                      "dataset_profile_sha256": file_sha256(contracts.dataset_profile_path) if contracts.dataset_profile_path else None,
                      "dataset_sha256": contracts.dataset.artifact.sha256 if contracts.dataset else None,
                      "checkpoint_sha256": contracts.manifest.checkpoint.sha256 if contracts.manifest.checkpoint else None, "seed": config.experiment.seed}
        package_root = Path(__file__).resolve().parents[1]
        provenance["core_source_sha256"] = {item.relative_to(package_root).as_posix(): file_sha256(item) for item in sorted(package_root.rglob("*.py"))}
        lock = package_root.parent.parent / "uv.lock"
        provenance["runtime_lock_sha256"] = file_sha256(lock) if lock.is_file() else None
        write_json(directory / "config.json", config.model_dump(mode="json"))
        write_json(directory / "profiles.json", {"model": contracts.manifest.model_dump(mode="json"), "robot": contracts.robot.model_dump(mode="json"), "dataset": contracts.dataset.model_dump(mode="json") if contracts.dataset else None})
        write_json(directory / "software-inventory.json", {"packages": sorted([{"name": item.metadata.get("Name", "unknown"), "version": item.version} for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower())})
        if random_state is not None:
            np.random.seed(config.experiment.seed)
        with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
            trials = [_Trial(run_id, repeat, config, contracts, None, directory, trace, stop_signal=stop_signal) for repeat in range(config.experiment.repeats)]
            for trial in trials:
                trial.lifecycle.transition(TrialState.PREFLIGHT)
            try:
                if admission_error is not None:
                    raise admission_error
                dataset_path = _preflight(config, contracts)
                source = Hdf5Dataset(dataset_path, contracts.dataset)
            except Exception as exc:
                preflight_error = exc
                results = []
                for trial in trials:
                    trial.block(exc)
                    results.append(trial.finish())
            else:
                for trial in trials:
                    trial.source = source
                    trial.lifecycle.transition(TrialState.PREPARED)
                try:
                    model = build_policy(contracts) if stop_signal.request_info is None else None
                except Exception as exc:
                    results = []
                    for trial in trials:
                        trial.fail(exc)
                        results.append(trial.finish())
                except KeyboardInterrupt:
                    stop_signal.request("interrupted")
                    results = []
                    for trial in trials:
                        trial.fail(StopRequested("interrupted"))
                        results.append(trial.finish())
                else:
                    if model is None:
                        results = []
                    elif config.execution.strategy == "sync":
                        buffer = ActionBuffer(contracts.manifest, config.execution)
                        results = []
                        for trial in trials:
                            trial.begin()
                            results.append(_sync_trial(trial, model, buffer))
                            if stop_signal.request_info is not None:
                                break
                    else:
                        backend = LocalPolicyBackend(model)
                        try:
                            results = asyncio.run(_async_trials(trials, backend, config, contracts))
                        except KeyboardInterrupt:
                            stop_signal.request("interrupted")
                            results = []
                            for trial in trials:
                                if trial._result is None:
                                    trial.fail(StopRequested("interrupted"))
                                results.append(trial.finish())
            not_started = []
            completed_ids = {item["trial_id"] for item in results}
            for trial in trials:
                if trial.trial_id not in completed_ids:
                    reason = stop_signal.request_info["reason"] if stop_signal.request_info else "prior_inference_not_drained"
                    trial.block(UnsupportedCapability(f"trial was not started: {reason}"))
                    item = trial.finish()
                    not_started.append({**item, "status": "NOT_RUN", "reason": reason})
        statuses = {item["status"] for item in results}
        status = "FAULT" if "FAULT" in statuses else "BLOCKED" if "BLOCKED" in statuses else "STOPPED" if "STOPPED" in statuses or not_started else "COMPLETED"
        task = resolve_task(config, contracts.manifest)
        report = {"schema_version": 2, "scope": "experiment_preflight" if preflight_error else "dataset_direct_record_only", "run_id": run_id,
                  "status": status, "environment_kind": config.environment.kind, "execution_mode": config.execution.mode,
                  "artifacts": str(directory), "dataset_source_kind": contracts.dataset.source_kind if contracts.dataset else None,
                  "timebase_source": source.timebase_source if source else None, "execution_strategy": config.execution.strategy,
                  "control_timebase": "logical_control_ticks" if config.execution.strategy == "sync" else "paced_local_monotonic_control_ticks",
                  "provenance": provenance, "trials": results, "requested_repeats": config.experiment.repeats,
                  "not_started_trials": not_started, "task": task.model_dump(mode="json"),
                  "conditions": {"policy_id": contracts.manifest.model_id, "policy_options": config.policy.options,
                                 "robot_profile_id": contracts.robot.profile_id,
                                 "execution": config.execution.model_dump(mode="json"),
                                 "effective_timeout_s": min(config.experiment.max_duration_s, task.timeout_s or config.experiment.max_duration_s)},
                  "not_run": ["genuine_training_loader_parity", "dataset_wire", "actuator_feedback", "task_success", "hardware_safety"]}
    finally:
        try:
            if source is not None:
                source.close()
        except Exception as exc:
            cleanup_errors.append({"resource": "dataset", "type": type(exc).__name__, "message": str(exc)})
        if model is not None:
            close_future = None
            try:
                if backend is not None:
                    close_future = backend.close()
                    close_future.result(timeout=1)
                else:
                    model.close()
            except Exception as exc:
                cleanup_errors.append({"resource": "policy", "type": type(exc).__name__, "message": str(exc)})
                if close_future is not None and not close_future.done() and lease is not None:
                    lease.defer_until_closed(close_future, random_state)
        if random_state is not None and (lease is None or not lease.deferred):
            np.random.set_state(random_state)
    if report is not None:
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "FAULT"
        provenance["core_source_sha256_after"] = {item.relative_to(package_root).as_posix(): file_sha256(item) for item in sorted(package_root.rglob("*.py"))}
        provenance["input_sha256_after"] = {key: file_sha256(item) if item is not None and item.is_file() else None for key, item in (
            ("configuration_sha256", contracts.source_path), ("manifest_sha256", contracts.manifest_path),
            ("robot_profile_sha256", contracts.robot_path), ("dataset_profile_sha256", contracts.dataset_profile_path))}
        provenance["runtime_lock_sha256_after"] = file_sha256(lock) if lock.is_file() else None
        report["source_stable"] = (provenance["core_source_sha256"] == provenance["core_source_sha256_after"] and
                                   provenance["runtime_lock_sha256"] == provenance["runtime_lock_sha256_after"] and
                                   all(provenance[key] == digest for key, digest in provenance["input_sha256_after"].items()))
        report["evidence_errors"] = [] if report["source_stable"] else [{"type": "SourceChanged", "message": "runtime source/config/profile/lock bytes changed during evaluation"}]
        if report["evidence_errors"]:
            report["status"] = "FAULT"
        report["policy_cleanup_pending"] = bool(lease is not None and lease.deferred)
        report["evidence_sha256"] = {item.name: file_sha256(item) for item in sorted(directory.iterdir()) if item.is_file()}
        write_json(directory / "results.json", report)
    if preflight_error is not None:
        preflight_error.report_path = str(directory / "results.json")
        raise preflight_error
    return report


def run_experiment(path: str | Path, *, stop_signal: StopSignal | None = None) -> dict:
    lease = _EvaluationLease()
    try:
        error = None if lease.acquired else UnsupportedCapability("another evaluation or pending model call still owns this process's global model/RNG state")
        return _run_experiment(path, stop_signal=stop_signal, lease=lease, admission_error=error)
    finally:
        if lease.acquired and not lease.deferred:
            _EVALUATION_LOCK.release()


class ExperimentRunner:
    """One run per instance; stop() requests host-side scheduling shutdown."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.stop_signal = StopSignal()
        self._lock = threading.Lock()
        self._used, self._report, self._error = False, None, None

    def stop(self, reason="operator_stop"):
        return self.stop_signal.request(reason)

    def evaluate(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this ExperimentRunner already owns an active evaluation")
        try:
            if self._used:
                if self._error is not None:
                    raise self._error
                return self._report
            self._used = True
            try:
                self._report = run_experiment(self.path, stop_signal=self.stop_signal)
                return self._report
            except BaseException as exc:
                self._error = exc
                raise
        finally:
            self._lock.release()
