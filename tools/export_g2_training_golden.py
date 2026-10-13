"""Export samples through the original StarVLA G2 training DataLoader.

Run in the existing model environment. Vendor code/weights are read locally,
never copied into the package. A private metadata copy isolates loader caches.
This exports the DataLoader boundary, not the VLM processor or model forward.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

from XPolicyLab.utils.process_data import decode_image_bit
from xpolicy.core.artifacts import file_sha256, source_revision, write_json
from xpolicy.diagnostics.golden import compare_bundles, save_bundle


REGISTRY = "examples/realRobots/AgiBotG2/train_files/data_registry/data_config.py"


def hashes(paths):
    return {str(path): file_sha256(path) for path in sorted(set(paths)) if path.is_file()}


def repository_identity(path):
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        return {"git_sha": git("rev-parse", "HEAD"), "worktree_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "worktree_dirty": None}


def export(source: Path, dataset_path: Path, checkpoint: Path, output: Path):
    source, dataset_path, checkpoint, output = (path.expanduser().resolve() for path in (source, dataset_path, checkpoint, output))
    config_path, weights = checkpoint / "config.full.yaml", checkpoint / "final_model/pytorch_model.pt"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["datasets"]["vla_data"]
    if data_cfg.get("action_mode") != "abs" or data_cfg.get("include_state") is not True or data_cfg.get("lerobot_version") != "v2.0":
        raise ValueError("this exporter requires the G2 absolute-action v2.0 state-enabled training configuration")
    expected_dataset = Path(data_cfg["data_root_dir"]).expanduser().resolve() / dataset_path.name
    if expected_dataset != dataset_path:
        raise ValueError("dataset path differs from the checkpoint training configuration")
    if not weights.is_file():
        raise ValueError("final_model/pytorch_model.pt is missing")
    if any(output == path or output in path.parents or path in output.parents for path in (source, dataset_path, checkpoint)):
        raise ValueError("output must be separate from the original loader, dataset and checkpoint trees")
    # Snapshot the implementation and the original metadata, not a mutable cache.
    protected = list((source / "starVLA").rglob("*.py")) + [source / REGISTRY]
    protected += [path for path in (dataset_path / "meta").rglob("*") if path.suffix in {".json", ".jsonl"}]
    protected += [config_path, checkpoint / "config.yaml", checkpoint / "dataset_statistics.json"]
    protected += [Path(__file__).resolve(), Path(inspect.getfile(compare_bundles)), Path(inspect.getfile(decode_image_bit))]
    from XPolicyLab.policy.StarVLA.model import Model
    from XPolicyLab.policy.StarVLA.image_preprocessing import resize_rgb

    protected += [Path(inspect.getfile(Model)), Path(inspect.getfile(resize_rgb))]
    before = hashes(protected)
    weights_sha = file_sha256(weights)
    output.mkdir(parents=True, exist_ok=False)
    view = output / "dataset-view" / dataset_path.name
    view.mkdir(parents=True)
    shutil.copytree(dataset_path / "meta", view / "meta", ignore=shutil.ignore_patterns("*.pkl", "*.tmp"))
    for name in ("data", "videos"):
        (view / name).symlink_to(dataset_path / name, target_is_directory=True)

    # Import the actual registry class, bypassing unrelated policy discovery.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import torch

    spec = importlib.util.spec_from_file_location("xpl_g2_training_registry", source / REGISTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if dataset_path.name != module.DATASET_NAME or data_cfg.get("data_mix") != module.MIXTURE_NAME:
        raise ValueError("dataset/data_mix differs from the imported G2 registry")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    factory = module.AgiBotG2DataConfig()
    dataset = factory.make_dataset(dataset_path=view, modality_configs=factory.modality_config(),
                                   transforms=factory.transform(), embodiment_tag=factory.embodiment_tag,
                                   video_backend=data_cfg["video_backend"], delete_pause_frame=False,
                                   data_cfg=data_cfg, dataset_name=module.DATASET_NAME)
    info = json.loads((view / "meta/info.json").read_text())
    modality = json.loads((view / "meta/modality.json").read_text())
    names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    model = config["framework"]["action_model"]
    if len(names) != model["state_dim"] or len(action_names) != model["action_dim"]:
        raise ValueError("source named field dimensions differ from checkpoint configuration")
    for kind, fields in (("state", names), ("action", action_names)):
        indices = [index for group in module.STATE_ACTION_FIELDS for index in range(modality[kind][group]["start"], modality[kind][group]["end"])]
        if indices != list(range(len(fields))):
            raise ValueError(f"training {kind} group concatenation differs from source named order")
    saved_statistics = json.loads((checkpoint / "dataset_statistics.json").read_text())[module.ROBOT_TYPE]
    stat_checks = []
    for kind in ("state", "action"):
        group_statistics = getattr(dataset.metadata.statistics, kind)
        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
            actual = np.concatenate([np.asarray(getattr(group_statistics[group], statistic)) for group in module.STATE_ACTION_FIELDS])
            saved = np.asarray(saved_statistics[kind][statistic])
            match = actual.shape == saved.shape and np.array_equal(actual, saved)
            within_tolerance = actual.shape == saved.shape and np.allclose(actual, saved, atol=1e-12, rtol=1e-7)
            stat_checks.append({"modality": kind, "statistic": statistic, "status": "PASS" if within_tolerance else "FAIL", "exact_status": "PASS" if match else "FAIL",
                                "max_absolute_error": float(np.max(np.abs(actual - saved))) if actual.shape == saved.shape else None})
    first_episode = int(dataset.all_steps[0][0])
    indices = [index for index, step in enumerate(dataset.all_steps) if int(step[0]) == first_episode]
    selected = sorted({indices[0], indices[len(indices) // 2], indices[-1]})
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected), batch_size=1, num_workers=0, collate_fn=list)
    camera_names = [key.removeprefix("video.") for key in factory.video_keys]
    # Unit metadata is absent in this source. Preserve that absence. These
    # samples cannot constitute an approved physical RobotProfile.
    raw_metadata = {"camera_order": camera_names, "color_space": "RGB", "state_names": names,
                    "state_units": None, "action_names": action_names, "action_units": None,
                    "state_space": "recorded_raw", "action_space": "recorded_absolute",
                    "timebase_source": "parquet_timestamp", "instruction_source": factory.language_keys[0]}
    provenance = {"xpolicylab": source_revision(), "training_repository": repository_identity(source),
                  "checkpoint": {"path": str(weights), "sha256": weights_sha},
                  "training_files_sha256": before, "seed": seed, "num_workers": 0,
                  "dataset_class": f"{type(dataset).__module__}.{type(dataset).__name__}",
                  "video_backend": data_cfg["video_backend"], "split": data_cfg["episode_split"]}
    reports, inputs = [], {}
    for index in selected:
        episode = int(dataset.all_steps[index][0])
        chunk = dataset.get_episode_chunk(episode)
        episode_path = view / dataset.data_path_pattern.format(episode_chunk=chunk, episode_index=episode)
        inputs.update(hashes([episode_path.resolve()] + [Path(dataset.get_video_path(episode, camera)).resolve() for camera in camera_names]))
    for index, batch in zip(selected, loader):
        packed = batch[0]
        episode, frame = (int(x) for x in dataset.all_steps[index])
        raw = dataset.get_step_data(episode, frame)
        table = dataset.get_trajectory_data(episode)
        chunk = dataset.get_episode_chunk(episode)
        episode_path = view / dataset.data_path_pattern.format(episode_chunk=chunk, episode_index=episode)
        source_files = [episode_path.resolve()] + [Path(dataset.get_video_path(episode, camera)).resolve() for camera in camera_names]
        frame_hashes = {str(path): inputs[str(path)] for path in source_files}
        sample = {"dataset_id": module.DATASET_NAME, "episode_id": episode,
                  "frame_id": int(table.iloc[frame]["frame_index"]), "timestamp_s": float(table.iloc[frame]["timestamp"]),
                  "clock_domain": "dataset_recorded", "loader_index": index}
        raw_arrays = {f"image/{name}": np.asarray(raw[key][0]) for name, key in zip(camera_names, factory.video_keys)}
        raw_arrays["state"] = np.concatenate([raw[key] for key in factory.state_keys], axis=1)
        raw_arrays["action_labels"] = np.concatenate([raw[key] for key in factory.action_keys], axis=1)
        raw_metadata_sample = {**raw_metadata, "instruction": str(packed["lang"])}
        axes = {"state": {"1": names}, "action_labels": {"1": action_names}}
        tensor_arrays = {f"image/{name}": np.asarray(image) for name, image in zip(camera_names, packed["image"])}
        tensor_arrays.update(state=np.asarray(packed["state"]), action_labels=np.asarray(packed["action"]), action_padding_mask=np.asarray(packed["action_padding_mask"]))
        if tensor_arrays["action_labels"].shape != (model["action_horizon"], len(action_names)):
            raise ValueError("packed training horizon/dimension differs from checkpoint")
        stem = output / f"episode-{episode:06d}-frame-{frame:06d}"
        golden = save_bundle(stem / "training", scope="genuine_training_loader_sample", sample=sample,
                             provenance={**provenance, "sample_inputs_sha256": frame_hashes}, layers=[
                                 {"name": "O1", "metadata": raw_metadata_sample, "arrays": raw_arrays, "axes": axes},
                                 {"name": "O2", "metadata": {"camera_order": camera_names, "state_names": names, "action_names": action_names,
                                                              "normalization": "G2StateActionTransform/StaticAwareQ99Normalizer",
                                                              "boundary": "DataLoader_packed_sample_before_VLM_processor", "instruction": str(packed["lang"])},
                                  "arrays": tensor_arrays, "axes": axes}])
        # Raw decoded pixels stay lossless. Never re-encode them with JPEG
        # when testing exact correspondence to the training video decoder.
        with h5py.File(stem / "raw-sample.hdf5", "w") as file:
            for name, array in raw_arrays.items():
                file.create_dataset(name, data=array[None] if name.startswith("image/") else array)
        with h5py.File(stem / "raw-sample.hdf5", "r") as file:
            recovered = {name: decode_image_bit(file[name][0]) if name.startswith("image/") else file[name][()] for name in raw_arrays}
        roundtrip = save_bundle(stem / "hdf5-roundtrip", scope="lossless_HDF5_shared_decoder_roundtrip", sample=sample,
                                provenance={"hdf5_sha256": file_sha256(stem / "raw-sample.hdf5")},
                                layers=[{"name": "O1", "metadata": raw_metadata_sample, "arrays": recovered, "axes": axes}])
        diff = compare_bundles(golden, roundtrip, layers=["O1"])
        write_json(stem / "roundtrip-diff.json", diff)
        # An independently computed resize counterexample describes the
        # existing adapter's INTER_AREA choice; it is not a model invocation.
        import cv2

        resize_candidate = {**tensor_arrays, **{f"image/{name}": cv2.resize(raw_arrays[f"image/{name}"], (224, 224), interpolation=cv2.INTER_AREA) for name in camera_names}}
        counterexample = save_bundle(stem / "opencv-resize-counterexample", scope="existing_resize_algorithm_counterexample", sample=sample,
                                     provenance={"algorithm": "cv2.INTER_AREA", "policy_invoked": False}, layers=[
                                         {"name": "O1", "metadata": raw_metadata_sample, "arrays": raw_arrays, "axes": axes},
                                         {"name": "O2", "metadata": json.loads((golden / "manifest.json").read_text())["layers"][1]["metadata"],
                                          "arrays": resize_candidate, "axes": axes}])
        resize_diff = compare_bundles(golden, counterexample)
        write_json(stem / "resize-diff.json", resize_diff)
        # Compare only the images actually produced by Model._convert_obs.
        # State/actions from the loader are not copied into this candidate.
        image_metadata = {"camera_order": camera_names, "color_space": "RGB", "instruction": str(packed["lang"]), "boundary": "packed_RGB_images_before_VLM_processor"}
        image_golden = save_bundle(stem / "training-images", scope="genuine_training_loader_image_boundary", sample=sample,
                                   provenance={**provenance, "sample_inputs_sha256": frame_hashes}, layers=[
                                       {"name": "O2", "metadata": image_metadata, "arrays": {f"image/{name}": tensor_arrays[f"image/{name}"] for name in camera_names}}])
        image_adapter = Model.__new__(Model)
        image_adapter.image_size, image_adapter.include_state = (224, 224), False
        image_adapter.model_cfg = {"image_resize_backend": "pillow_bicubic"}
        aliases = dict(zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), camera_names))
        observation = {"vision": {alias: {"color": raw_arrays[f"image/{name}"]} for alias, name in aliases.items()}, "instruction": str(packed["lang"])}
        converted = image_adapter._convert_obs(observation)
        adapter_images = save_bundle(stem / "adapter-pillow-images", scope="actual_StarVLA_image_preprocessing_only", sample=sample,
                                      provenance={"policy_method": "Model._convert_obs", "model_constructed": False, "include_state": False, "camera_aliases": aliases, "image_resize_backend": "pillow_bicubic"}, layers=[
                                          {"name": "O2", "metadata": {**image_metadata, "instruction": converted["lang"]}, "arrays": {f"image/{name}": image for name, image in zip(camera_names, converted["image"])}}])
        image_diff = compare_bundles(image_golden, adapter_images)
        write_json(stem / "adapter-image-diff.json", image_diff)
        reports.append({"sample": sample, "path": str(stem), "roundtrip": diff["status"],
                        "opencv_resize": resize_diff["status"], "resize_first_divergence": resize_diff["first_divergence"],
                        "actual_adapter_pillow_images": image_diff["status"],
                        "padding_steps": int(np.count_nonzero(packed["action_padding_mask"]))})
    after, inputs_after = hashes(protected), hashes(Path(path) for path in inputs)
    # Verify checkpoint bytes also remained stable throughout the export.
    unchanged = before == after and inputs == inputs_after and weights_sha == file_sha256(weights)
    report = {"schema_version": 1, "scope": "genuine_training_loader_export", "status": "PASS" if unchanged and all(item["roundtrip"] == "PASS" and item["actual_adapter_pillow_images"] == "PASS" for item in reports) else "FAIL",
              "provenance": provenance, "protected_files_sha256_after": after, "sample_inputs_sha256": inputs,
              "sample_inputs_sha256_after": inputs_after, "source_stable": unchanged,
              "dataset": {"episodes": info["total_episodes"], "frames": info["total_frames"], "training_steps": len(dataset), "state_dim": len(names), "action_dim": len(action_names)},
              "samples": reports, "not_run": ["VLM_processor_tensor_parity", "checkpoint_forward", "dataset_wire", "physical_units_and_GDK_mapping"],
              "checkpoint_statistics_alignment": {"scope": "saved_statistics_vs_current_loader_named_order", "status": "PASS" if all(item["status"] == "PASS" for item in stat_checks) else "FAIL",
                                                  "exact_status": "PASS" if all(item["exact_status"] == "PASS" for item in stat_checks) else "FAIL", "tolerance": {"atol": 1e-12, "rtol": 1e-7},
                                                  "q99_parameters_exact_status": "PASS" if all(item["exact_status"] == "PASS" for item in stat_checks if item["statistic"] in {"q01", "q99"}) else "FAIL", "checks": stat_checks},
              "limitations": ["Unit metadata is absent; null units are not approved physical calibration.",
                              "The current training repository may differ from the checkpoint's historical training source; current file hashes are recorded.",
                              "O2 is the actual packed DataLoader sample, before the model's VLM processor.",
                              "Actual adapter parity covers images only; the 24D state/action adapter mapping and final model processor remain unverified.",
                              "OpenCV resize counterexamples do not execute a policy adapter or checkpoint."]}
    inventory = {"python": sys.version, "executable": sys.executable, "packages": sorted([{"name": package.metadata.get("Name", "unknown"), "version": package.version} for package in importlib.metadata.distributions()], key=lambda item: (item["name"], item["version"]))}
    write_json(output / "software-inventory.json", inventory)
    report["software_inventory_sha256"] = file_sha256(output / "software-inventory.json")
    write_json(output / "TRAINING_GOLDEN_REPORT.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    new_output = not args.output.exists()
    try:
        report = export(args.source, args.dataset, args.checkpoint, args.output)
    except Exception as exc:
        # Keep the nonzero exit and logs; do not leave a partial export looking
        # like a successful run. Never overwrite an existing report on retry.
        if new_output and args.output.is_dir() and not (args.output / "TRAINING_GOLDEN_REPORT.json").exists():
            write_json(args.output / "TRAINING_GOLDEN_REPORT.json", {"schema_version": 1, "scope": "genuine_training_loader_export", "status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}})
        raise
    print({"report": str(args.output / "TRAINING_GOLDEN_REPORT.json"), "status": report["status"]})
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
