"""Create a small synthetic XPL HDF5 trajectory and an offline experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from XPolicyLab.utils.process_data import encode_image_bit
from xpolicy.core.artifacts import file_sha256
from xpolicy.core.config import load_service

ROOT = Path(__file__).resolve().parents[1]


def create_fixture(output: Path, frames=5, strategy="sync", *, with_suite=False):
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")
    if strategy not in {"sync", "async_latest_only"}:
        raise ValueError("unsupported fixture execution strategy")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, contracts = load_service(ROOT / "configs/policies/mock.yaml")
    data = output / "trajectory.hdf5"
    with h5py.File(data, "w") as file:
        arm = np.linspace(0.25, 0.5, frames, dtype=np.float32)[:, None]
        gripper = np.linspace(0.02, 0.04, frames, dtype=np.float32)[:, None]
        file.create_dataset("state/joint_states", data=arm)
        file.create_dataset("state/gripper_states", data=gripper)
        file.create_dataset("action/joint_states", data=arm + 0.1)
        file.create_dataset("action/gripper_states", data=gripper + 0.001)
        file.create_dataset("timestamps", data=np.arange(frames) / contracts.robot.control_hz)
        file.create_dataset("frame_ids", data=np.arange(frames, dtype=np.int64))
        file.create_dataset("instruction", data=json.dumps(["hold the synthetic pose"]))
        images = file.create_dataset("vision/cam_head/colors", (frames,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        image = np.empty((8, 8, 3), dtype=np.uint8)
        image[:, :, 0], image[:, :, 1], image[:, :, 2] = 210, 50, 15
        for index in range(frames):
            # Container conversion only: the encoded RGB marker and pixels
            # are owned by encode_image_bit, not by this HDF5 writer.
            images[index] = np.asarray(bytearray(encode_image_bit(image)), dtype=np.uint8)
    manifest = contracts.manifest.model_dump(mode="json")
    boundary = {key: manifest[key] for key in ("schema_version", "cameras", "state_fields", "action_fields", "action_type", "reference_frame", "quaternion_order")}
    boundary["cameras"][0]["source_key"] = "vision/cam_head/colors"
    profile = {**boundary, "dataset_id": "synthetic_mock_hdf5_v1", "robot_id": contracts.robot.robot_id,
               "source_kind": "synthetic", "artifact": {"uri": "trajectory.hdf5", "sha256": file_sha256(data)},
               "decoder_id": "xpl_hdf5_v1", "sample_hz": contracts.robot.control_hz, "clock_domain": "synthetic_recording",
               "hdf5": {"state": {"arm_joint": {"key": "state/joint_states", "column": 0}, "gripper": {"key": "state/gripper_states", "column": 0}},
                        "action": {"arm_joint": {"key": "action/joint_states", "column": 0}, "gripper": {"key": "action/gripper_states", "column": 0}},
                        "timestamp_key": "timestamps", "frame_id_key": "frame_ids", "instruction_key": "instruction"}}
    experiment = {"schema_version": 1, "experiment": {"name": "synthetic_dataset_replay", "seed": 42, "repeats": 2, "max_duration_s": 10},
                  "task": {"task_id": "synthetic_hold", "allowed_action_types": ["joint_position"],
                           "instruction": {"source": "dataset"}, "reset": "recorded_episode_start",
                           "evaluator": {"source": "not_measured"}},
                  "policy": {"name": "mock", "api": "chunk", "manifest": str(contracts.manifest_path)},
                  "environment": {"kind": "dataset_direct", "robot": str(contracts.robot_path), "dataset_profile": "dataset.yaml", "dataset_path": "trajectory.hdf5"},
                  "execution": {"mode": "record_only", "strategy": strategy, "control_hz": contracts.robot.control_hz, "execute_horizon": 2,
                                "latency_trim": strategy == "async_latest_only", "overlap_blend_steps": 0, "empty_buffer": "skip_record"},
                  "artifacts": {"root": "results"}}
    for name, value in (("dataset.yaml", profile), ("experiment.yaml", experiment)):
        (output / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    if with_suite:
        alternate = copy.deepcopy(experiment)
        other = "async_latest_only" if strategy == "sync" else "sync"
        alternate["execution"]["strategy"] = other
        alternate["execution"]["latency_trim"] = other == "async_latest_only"
        (output / "alternate.yaml").write_text(yaml.safe_dump(alternate, sort_keys=False), encoding="utf-8")
        suite = {"schema_version": 1, "name": "synthetic_strategy_seed_suite", "artifacts_root": "suite-results",
                 "process_timeout_s": 30., "cleanup_grace_s": 2., "fail_fast": False,
                 "cases": [{"case_id": strategy, "config": "experiment.yaml", "seeds": [3, 7], "repeats": 1},
                           {"case_id": other, "config": "alternate.yaml", "seeds": [3, 7], "repeats": 1}]}
        (output / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    return output / "experiment.yaml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--strategy", choices=("sync", "async_latest_only"), default="sync")
    parser.add_argument("--with-suite", action="store_true", help="also write a two-strategy, two-seed suite")
    args = parser.parse_args(argv)
    print(create_fixture(args.output, args.frames, args.strategy, with_suite=args.with_suite))


if __name__ == "__main__":
    main()
