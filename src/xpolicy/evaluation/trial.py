"""Trial state, task constraints and evidence shared by Runner strategies."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

import numpy as np

from xpolicy.core.contracts import ContractValidator
from xpolicy.core.tasks import resolve_task
from xpolicy.evaluation.lifecycle import StopRequested, StopSignal, TrialLifecycle, TrialState
from xpolicy.evaluation.metrics import process_cpu_seconds, series


class _Trial:
    def __init__(self, run_id, repeat, config, contracts, source, directory, trace, *, stop_signal=None):
        self.run_id, self.trial_id = run_id, f"{run_id}:{repeat}"
        self.config, self.contracts, self.source, self.directory, self.trace = config, contracts, source, directory, trace
        self.repeat, self.generation, self.row = repeat, repeat + 1, 0
        self.started = time.monotonic()
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
            if self.source is not None and callable(getattr(self.source, "stop", None)):
                self.source.stop(reason)
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
        obs = self.source.observe(row)
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
        receipt = self.source.record(commands, control_tick=control_row)
        accepted, labels = receipt.values, receipt.labels
        self.recorded += len(commands)
        self.event("record_only_submission", observation, **commands[0].ticket.context(),
                   control_frame_id=receipt.control_frame_ids[0], control_tick=control_row,
                   control_frame_ids=receipt.control_frame_ids, control_timestamps_s=receipt.control_timestamps_s,
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
