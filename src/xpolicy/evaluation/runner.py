"""Noncausal dataset replay through the shared record-only execution owner."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import platform
import sys
import time
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np

from xpolicy.core.artifacts import file_sha256, source_revision, write_json
from xpolicy.core.config import load_experiment
from xpolicy.core.contracts import ContractValidator
from xpolicy.core.errors import UnsupportedCapability
from xpolicy.core.tasks import resolve_task, validate_task
from xpolicy.environment.contracts import EpisodeContext
from xpolicy.environment.registry import environment_registry
from xpolicy.execution.async_latest import AsyncLatestOnly, LocalPolicyBackend
from xpolicy.execution.buffer import ActionBuffer
from xpolicy.policy.runtime import build_policy
from xpolicy.evaluation.lifecycle import StopRequested, StopSignal, TrialLifecycle, TrialState
from xpolicy.evaluation.trial import _Trial


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
    entry = environment_registry().require(config.environment.kind)
    if entry.runtime_contract != "recorded_timeline_v1" or config.execution.mode not in entry.capabilities or config.execution.strategy not in entry.capabilities:
        raise UnsupportedCapability("Runner requires an implemented environment with compatible execution capabilities")
    if config.policy.api != "chunk":
        raise UnsupportedCapability("dataset replay requires an explicit complete-chunk policy")
    ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
    ActionBuffer(contracts.manifest, config.execution)
    return entry.load().preflight(config, contracts)



def _sync_trial(trial, model, buffer):
    buffer.emit = trial.execution_event
    trial.generation = buffer.reset(trial.trial_id)
    try:
        trial.check_budget()
        receipt = trial.source.reset(EpisodeContext(trial.trial_id, trial.generation, trial.config.experiment.seed, trial.repeat))
        trial.event("environment_reset", receipt=receipt)
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
                _, now_s, clock_domain = trial.source.time_at(control_row)
                command = buffer.take(control_row, now_s=now_s, clock_domain=clock_domain)
                if command is not None:
                    commands.append(command)
                else:
                    raise RuntimeError("sync prediction did not provide an executable record-only step")
            trial.record(commands, obs, row)
            row += count
        trial.source.verify_evidence()
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
                receipt = trial.source.reset(EpisodeContext(trial.trial_id, trial.generation, config.experiment.seed, trial.repeat))
                trial.event("environment_reset", receipt=receipt)
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
                    now_s = trial.source.time_at(current_tick)[1] if current_tick < trial.source.length else None
                    return current_tick, now_s, trial.source.clock_domain

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
                    _, now_s, clock_domain = trial.source.time_at(actual_tick)
                    command = controller.take(actual_tick, now_s=now_s, clock_domain=clock_domain)
                    if command is not None:
                        trial.record([command], obs, actual_tick)
                    row = actual_tick + 1
                controller.stop("trial_end")
                trial.stop("fault" if trial.fault else "dataset_end")
                remaining = max(0, trial.timeout_s - (time.monotonic() - trial.started))
                await controller.wait_idle(remaining)
                if controller.fault is not None:
                    trial.capture_fault(controller.fault)
                trial.source.verify_evidence()
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
        write_json(directory / "software-inventory.json", {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
                   "platform": platform.platform(), "libc": list(platform.libc_ver()),
                   "packages": sorted([{"name": item.metadata.get("Name", "unknown"), "version": item.version} for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower())})
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
                source = environment_registry().create(config.environment.kind, config, contracts, dataset_path)
            except Exception as exc:
                preflight_error = exc
                results = []
                for trial in trials:
                    trial.block(exc)
                    results.append(trial.finish())
            else:
                for trial in trials:
                    trial.source = source
                    trial.event("environment_prepared", provider=source.describe())
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
                  "environment_provider": source.describe() if source is not None else None,
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
