"""Recorded environment: observations stay independent of submitted actions."""

import numpy as np

from xpolicy.core.artifacts import verify_artifact
from xpolicy.core.contracts import ContractValidator
from xpolicy.dataset.hdf5 import Hdf5Dataset
from xpolicy.execution.buffer import ScheduledAction, tick
from xpolicy.execution.record_only import RecordOnlySink
from xpolicy.environment.contracts import EpisodeContext, RecordedSubmission




class RecordedDatasetEnvironment:
    runtime_contract = "recorded_timeline_v1"
    causal_feedback = False

    @staticmethod
    def preflight(config, contracts):
        if config.environment.kind != "dataset_direct" or config.execution.mode != "record_only":
            raise ValueError("recorded dataset provider requires dataset_direct/record_only")
        if config.environment.episode != 0:
            raise ValueError("xpl_hdf5_v1 selects one trajectory file; episode must be 0")
        ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
        if contracts.dataset is None:
            raise ValueError("dataset_direct requires a dataset profile")
        if config.execution.control_hz != contracts.robot.control_hz or config.execution.control_hz != contracts.dataset.sample_hz:
            raise ValueError("robot control rate and dataset sample rate must agree; resampling is not implemented")
        path = verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
        if config.environment.dataset_path != path:
            raise ValueError("dataset_path must match the verified dataset artifact")
        return path

    def __init__(self, config, contracts, verified_path):
        # The provider owns the decoder/file handle and the record-only sink.
        # The Runner owns model calls, execution ticks and trial state.
        if verified_path != self.preflight(config, contracts):
            raise ValueError("environment path differs from verified preflight")
        self.contracts, self.profile = contracts, contracts.dataset
        self.clock_domain = self.profile.clock_domain
        self._source = Hdf5Dataset(verified_path, contracts.dataset)
        self.length, self.timebase_source = self._source.length, self._source.timebase_source
        self._context, self._closed, self._stopped = None, False, True
        self._sink = None

    def reset(self, context):
        if self._closed:
            raise RuntimeError("recorded environment is closed")
        if not isinstance(context, EpisodeContext):
            raise TypeError("environment reset requires an EpisodeContext")
        if self._context is not None and context.generation <= self._context.generation:
            raise ValueError("environment reset generation must increase")
        self._context, self._stopped, self._sink = context, False, RecordOnlySink()
        return {"episode_id": context.episode_id, "generation": context.generation,
                "reset": "recorded_episode_start", "physical_robot_reset": None,
                "physical_scene_reset": None, "device_ack": None}

    def _active(self):
        if self._closed or self._stopped or self._context is None:
            raise RuntimeError("recorded environment requires an active reset episode")

    def time_at(self, control_tick):
        if self._closed:
            raise RuntimeError("recorded environment is closed")
        control_tick = tick(control_tick)
        if control_tick >= self.length:
            raise IndexError("recorded control tick lies outside the episode")
        return int(self._source.frame_ids[control_tick]), float(self._source.timestamps[control_tick]), self.profile.clock_domain

    def observe(self, control_tick):
        self._active()
        self.time_at(control_tick)
        return self._source.observe(control_tick, self._context.episode_id)

    def record(self, commands, *, control_tick):
        self._active()
        commands = tuple(commands)
        if not commands or any(not isinstance(command, ScheduledAction) for command in commands):
            raise ValueError("environment recording requires scheduled commands")
        control_tick = tick(control_tick)
        frames, timestamps = [], []
        for offset, command in enumerate(commands):
            frame, timestamp, clock = self.time_at(control_tick + offset)
            if command.control_tick != control_tick + offset:
                raise ValueError("recorded submission must align with contiguous control ticks")
            context = command.ticket
            if context.start_control_tick > command.control_tick:
                raise ValueError("command cannot precede its source observation tick")
            if (context.episode_id, context.generation) != (self._context.episode_id, self._context.generation):
                raise ValueError("command does not belong to the active environment episode/generation")
            source_frame, _, _ = self.time_at(context.start_control_tick)
            if context.source_frame_id != source_frame or context.source_clock_domain != clock:
                raise ValueError("command source identity differs from the recorded timeline")
            values = np.asarray(command.values)
            if values.shape != (len(self.contracts.manifest.action_fields),) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError("recorded command values differ from the declared finite action layout")
            frames.append(frame); timestamps.append(timestamp)
        labels = self._source.labels(control_tick, len(commands))
        if labels is not None:
            labels = np.array(labels, copy=True)
            labels.setflags(write=False)
        values = self._sink.submit_commands(commands)
        return RecordedSubmission(values, labels, tuple(frames), tuple(timestamps))

    def stop(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("environment stop requires a nonempty reason")
        changed = not self._stopped
        self._stopped = True
        return changed

    def verify_evidence(self):
        if self._closed:
            raise RuntimeError("recorded environment is closed")
        return verify_artifact(self.contracts.dataset.artifact, self.contracts.dataset_profile_path)

    def describe(self):
        return {"kind": "dataset_direct", "runtime_contract": self.runtime_contract,
                "causal_feedback": False, "dataset_id": self.profile.dataset_id,
                "source_kind": self.profile.source_kind, "frames": self.length,
                "clock_domain": self.profile.clock_domain, "timebase_source": self.timebase_source,
                "device_driver": None, "task_success_source": "not_measured"}

    def close(self):
        if not self._closed:
            self.stop("close")
            self._source.close()
            self._closed = True
