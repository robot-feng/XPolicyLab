"""Synchronous, noncausal dataset replay with one chunk owner."""

from __future__ import annotations

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
from xpolicy.execution.record_only import RecordOnlySink
from xpolicy.policy.runtime import build_policy


def _preflight(config, contracts):
    if config.environment.kind != "dataset_direct":
        raise UnsupportedCapability("offline runner currently supports dataset_direct only")
    if config.execution.mode != "record_only" or config.execution.strategy != "sync":
        raise UnsupportedCapability("dataset replay requires record_only with sync execution")
    if config.policy.api != "chunk":
        raise UnsupportedCapability("dataset replay requires an explicit complete-chunk policy")
    if config.environment.episode != 0:
        raise ValueError("xpl_hdf5_v1 selects one trajectory file; episode must be 0")
    ContractValidator.require_profiles(contracts.manifest, contracts.robot, contracts.dataset)
    if contracts.dataset is None:
        raise ValueError("dataset_direct requires a dataset profile")
    if config.execution.execute_horizon > contracts.manifest.predicted_horizon:
        raise ValueError("execute_horizon exceeds the model prediction horizon")
    if config.execution.control_hz != contracts.robot.control_hz or config.execution.control_hz != contracts.dataset.sample_hz:
        raise ValueError("robot control rate and dataset sample rate must agree; resampling is not implemented")
    dataset_path = verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
    if config.environment.dataset_path != dataset_path:
        raise ValueError("dataset_path must match the verified dataset artifact")
    return dataset_path


def run_experiment(path: str | Path) -> dict:
    config, contracts = load_experiment(path)
    dataset_path = _preflight(config, contracts)
    source = Hdf5Dataset(dataset_path, contracts.dataset)
    model = None
    try:
        # Stream layout checks happen before model construction.
        model = build_policy(contracts)
        run_id = uuid4().hex
        directory = config.artifacts.root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        provenance = {
            **source_revision(), "configuration_sha256": file_sha256(contracts.source_path),
            "manifest_sha256": file_sha256(contracts.manifest_path),
            "robot_profile_sha256": file_sha256(contracts.robot_path),
            "dataset_profile_sha256": file_sha256(contracts.dataset_profile_path),
            "dataset_sha256": contracts.dataset.artifact.sha256,
            "checkpoint_sha256": contracts.manifest.checkpoint.sha256 if contracts.manifest.checkpoint else None,
            "seed": config.experiment.seed,
        }
        package_root = Path(__file__).resolve().parents[1]
        provenance["core_source_sha256"] = {item.relative_to(package_root).as_posix(): file_sha256(item) for item in sorted(package_root.rglob("*.py"))}
        write_json(directory / "config.json", config.model_dump(mode="json"))
        write_json(directory / "profiles.json", {"model": contracts.manifest.model_dump(mode="json"), "robot": contracts.robot.model_dump(mode="json"), "dataset": contracts.dataset.model_dump(mode="json")})
        write_json(directory / "software-inventory.json", {"packages": sorted([{"name": item.metadata.get("Name", "unknown"), "version": item.version} for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower())})
        trials = []
        np.random.seed(config.experiment.seed)
        with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
            for repeat in range(config.experiment.repeats):
                trial_id = f"{run_id}:{repeat}"
                sink, row, predictions = RecordOnlySink(), 0, 0
                started = time.monotonic()
                errors, fault = [], None

                def event(stage, obs=None, **payload):
                    identity = {"run_id": run_id, "trial_id": trial_id, "episode_id": trial_id,
                                "episode_generation": repeat, "monotonic_s": time.monotonic(),
                                "stage": stage, "source_row": row,
                                "frame_id": obs.frame_id if obs is not None else None}
                    trace.write(json.dumps({**identity, **payload}, allow_nan=False) + "\n")
                    trace.flush()

                try:
                    model.reset()
                    while row < source.length:
                        if time.monotonic() - started >= config.experiment.max_duration_s:
                            raise TimeoutError("trial exceeded max_duration_s between predictions")
                        obs = source.observe(row, trial_id)
                        ContractValidator.observation(obs, contracts.robot, contracts.manifest)
                        image_hashes = {name: hashlib.sha256(image.tobytes()).hexdigest() for name, image in obs.images.items()}
                        request_id, chunk_id = uuid4().hex, uuid4().hex
                        context = {"request_id": request_id, "chunk_id": chunk_id, "profile_id": contracts.robot.profile_id}
                        event("observation", obs, **context, timestamp_s=obs.timestamp_s, clock_domain=obs.clock_domain,
                              state=obs.state.tolist(), state_names=obs.state_names, state_units=obs.state_units,
                              images={name: {"sha256": image_hashes[name], "shape": image.shape, "color_space": "RGB"} for name, image in obs.images.items()})
                        if config.artifacts.save_images and config.validation.trace_level == "full":
                            np.savez_compressed(directory / f"input-{repeat}-{row}.npz", state=obs.state, **{f"image_{index}": image for index, image in enumerate(obs.images.values())})
                        before = time.monotonic()
                        chunk = model.predict_action_chunk(obs)
                        latency_s = time.monotonic() - before
                        predictions += 1
                        if time.monotonic() - started >= config.experiment.max_duration_s:
                            raise TimeoutError("prediction returned after max_duration_s; action was not recorded")
                        event("prediction", obs, **context, values=chunk.values.tolist() if config.artifacts.save_actions else None,
                              shape=chunk.values.shape, action_type=chunk.action_type.value, action_units=chunk.action_units,
                              joint_names=chunk.joint_names, sequence_id=chunk.sequence_id, latency_s=latency_s)
                        count = min(config.execution.execute_horizon, source.length - row)
                        labels = source.labels(row, count)
                        accepted = sink.submit(chunk, count)
                        event("record_only_submission", obs, **context, sequence_id=chunk.sequence_id,
                              values=accepted.tolist() if config.artifacts.save_actions else None, count=count,
                              causal=False, device_feedback=None)
                        if labels is not None:
                            errors.append(accepted.astype(np.float64) - labels.astype(np.float64))
                        row += count
                    verify_artifact(contracts.dataset.artifact, contracts.dataset_profile_path)
                except Exception as exc:
                    fault = {"type": type(exc).__name__, "message": str(exc)}
                    event("fault", fault=fault)
                trial = {"trial_id": trial_id, "status": "FAULT" if fault else "COMPLETED",
                         "fault": fault, "dataset_frames": source.length, "recorded_action_steps": row,
                         "predictions": predictions, "duration_s": time.monotonic() - started,
                         "task_success": None, "success_source": "not_measured", "causal": False}
                if errors:
                    error = np.concatenate(errors, axis=0)
                    trial["behavior_cloning_error"] = {
                        field.name: {"unit": field.unit, "mae": float(np.mean(np.abs(error[:, index]))),
                                     "rmse": float(np.sqrt(np.mean(error[:, index] ** 2))), "samples": len(error)}
                        for index, field in enumerate(contracts.manifest.action_fields)
                    }
                trials.append(trial)
                event("trial_end", result=trial)
        report = {"schema_version": 1, "scope": "dataset_direct_record_only", "run_id": run_id,
                  "status": "FAULT" if any(item["status"] == "FAULT" for item in trials) else "COMPLETED",
                  "artifacts": str(directory), "dataset_source_kind": contracts.dataset.source_kind,
                  "timebase_source": source.timebase_source, "provenance": provenance, "trials": trials,
                  "not_run": ["genuine_training_loader_parity", "dataset_wire", "actuator_feedback", "task_success", "hardware_safety"]}
        write_json(directory / "results.json", report)
        return report
    finally:
        source.close()
        if model is not None:
            model.close()
