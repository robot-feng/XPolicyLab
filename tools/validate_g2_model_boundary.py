"""Compare genuine G2 loader samples with XPL normalization and model inputs.

Uses the supplied upstream RollingFlow implementation and checkpoint, composed
with XPL's actual normalization wrapper. This is a diagnostic composition, not
a registered 24D hardware adapter. No EVA/SDK client or transport is loaded.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from xpolicy.core.artifacts import file_sha256, source_revision, write_json
from xpolicy.diagnostics.golden import compare_bundles, load_bundle, save_bundle


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "examples/realRobots/AgiBotG2/train_files/data_registry/data_config.py"
SERVER = ROOT / "policy/StarVLA/source_starvla/deployment/model_server"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(paths):
    return {str(path): file_sha256(path) for path in sorted(set(paths)) if path.is_file()}


def arrays(manifest, values, layer):
    item = next(item for item in manifest.layers if item.name == layer)
    return {name: values[desc.key] for name, desc in item.arrays.items()}


def bundle(path, manifest, values, metadata, *, name="O2", role="actual_boundary_capture"):
    names = manifest.layers[0].metadata["state_names"]
    action_names = manifest.layers[0].metadata["action_names"]
    axes = {"state": {"1": names}, "model/state": {"2": names},
            "action_labels": {"1": action_names},
            "normalized_actions": {"2": action_names}, "env_actions": {"2": action_names}}
    return save_bundle(path, scope="G2_model_boundary_diagnostic_composition", sample=manifest.sample,
                       provenance={"training_bundle_sha256": manifest.arrays_sha256, "role": role}, layers=[
                           {"name": name, "metadata": metadata, "arrays": values,
                            "axes": {key: axes[key] for key in values if key in axes}}])


def validate(args, report):
    source, dataset_path, checkpoint, golden, output = (path.expanduser().resolve() for path in
        (args.source, args.dataset, args.checkpoint, args.golden, args.output))
    for path in (source, dataset_path, checkpoint, golden):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("output must be separate from input trees")
    config = yaml.safe_load((checkpoint / "config.full.yaml").read_text())
    data_cfg = config["datasets"]["vla_data"]
    if config["framework"]["name"] != "QwenGR00TRollingFlow" or data_cfg.get("include_state") is not True or data_cfg.get("action_mode") != "abs":
        raise ValueError("requires a state-enabled absolute-action G2 RollingFlow checkpoint")
    if Path(data_cfg["data_root_dir"]).resolve() / dataset_path.name != dataset_path:
        raise ValueError("dataset does not match checkpoint data_root_dir")
    weights = checkpoint / "final_model/pytorch_model.pt"
    base = Path(config["framework"]["qwenvl"]["base_vlm"]).expanduser().resolve()
    protected = list((source / "starVLA").rglob("*.py")) + list((source / "deployment/model_server").rglob("*.py"))
    protected += [source / REGISTRY, weights] + list(checkpoint.glob("*.yaml")) + [checkpoint / "dataset_statistics.json"]
    protected += list(base.rglob("*")) if args.forward else []
    protected += list((dataset_path / "meta").glob("*.json*"))
    protected += list(golden.glob("episode-*/training/*")) + [golden / "TRAINING_GOLDEN_REPORT.json"]
    for manifest_path in golden.glob("episode-*/training/manifest.json"):
        original = json.loads(manifest_path.read_text())
        protected += [Path(path) for path in original["provenance"]["sample_inputs_sha256"]]
    protected += [Path(__file__), ROOT / "src/xpolicy/diagnostics/golden.py", ROOT / "utils/state_precision.py",
                  ROOT / "src/xpolicy/core/artifacts.py", ROOT / "utils/process_data.py",
                  ROOT / "pyproject.toml", ROOT / "uv.lock",
                  ROOT / "policy/StarVLA/model.py", ROOT / "policy/StarVLA/image_preprocessing.py",
                  SERVER / "policy_norm_processor.py", SERVER / "policy_wrapper.py"]
    before = snapshot(protected)
    report["provenance"] = {"xpolicylab": source_revision(), "source_file_sha256": before,
                            "checkpoint_sha256": before[str(weights)], "golden": str(golden)}
    write_json(output / "MODEL_BOUNDARY_REPORT.json", report)
    for manifest_path in golden.glob("episode-*/training/manifest.json"):
        original = json.loads(manifest_path.read_text())
        for path, digest in original["provenance"]["sample_inputs_sha256"].items():
            if before.get(str(Path(path))) != digest:
                raise ValueError(f"dataset sample differs from the original golden input: {path}")
        for path, digest in original["provenance"]["training_files_sha256"].items():
            if Path(path).is_relative_to(source) and before.get(str(Path(path))) != digest:
                raise ValueError(f"training implementation differs from the exported golden: {path}")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import torch
    from PIL import Image
    from XPolicyLab.policy.StarVLA.model import Model

    torch.set_num_threads(args.cpu_threads)
    registry = load_module("xpl_g2_boundary_registry", source / REGISTRY)
    if data_cfg["data_mix"] != registry.MIXTURE_NAME or dataset_path.name != registry.DATASET_NAME:
        raise ValueError("checkpoint mix/dataset differs from actual registry")
    view = output / "dataset-view" / dataset_path.name
    view.mkdir(parents=True)
    shutil.copytree(dataset_path / "meta", view / "meta", ignore=shutil.ignore_patterns("*.pkl", "*.tmp"))
    for name in ("data", "videos"):
        (view / name).symlink_to(dataset_path / name, target_is_directory=True)
    factory = registry.AgiBotG2DataConfig()
    dataset = factory.make_dataset(dataset_path=view, modality_configs=factory.modality_config(),
        transforms=factory.transform(), embodiment_tag=factory.embodiment_tag,
        video_backend=data_cfg["video_backend"], delete_pause_frame=False,
        data_cfg=data_cfg, dataset_name=registry.DATASET_NAME)
    norm_module = load_module("xpl_g2_boundary_norm", SERVER / "policy_norm_processor.py")
    wrapper_module = load_module("xpl_g2_boundary_wrapper", SERVER / "policy_wrapper.py")
    proc = norm_module.PolicyNormProcessor(str(weights), state_input_dtype=args.state_input_dtype,
                                          model_state_dtype=args.model_state_dtype)
    legacy_proc = norm_module.PolicyNormProcessor(str(weights))
    wrapper = wrapper_module.PolicyServerWrapper.__new__(wrapper_module.PolicyServerWrapper)
    wrapper._default_unnorm_key, wrapper._available_unnorm_keys = proc.unnorm_key, proc.available_unnorm_keys
    wrapper._norm_processors, wrapper._logged_state_contract = {proc.unnorm_key: proc}, True
    report["composition"] = {"normalizer": str(SERVER / "policy_norm_processor.py"),
        "wrapper": str(SERVER / "policy_wrapper.py"), "registry": str(source / REGISTRY),
        "state_input_dtype": args.state_input_dtype, "model_state_dtype": args.model_state_dtype,
        "full_adapter_constructor_used": False, "hardware_profile_registered": False,
        "state_keys": proc.state_keys, "state_key_dims": proc.state_key_dims,
        "action_keys": proc.action_keys, "action_key_dims": proc.action_key_dims}
    report["samples"] = []
    model, captured = None, {}
    model_calls = {"framework_predict": 0, "vlm_input_build": 0, "action_head_predict": 0}
    seed = int(config.get("seed", 42))

    def seed_rng():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if args.forward and str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    if args.forward:
        if not str(args.device).startswith("cuda"):
            raise ValueError("this checkpoint inference implementation requires a CUDA device")
        device = torch.device(args.device)
        free, total = torch.cuda.mem_get_info(device)
        cap = int(args.gpu_memory_gib * 1024**3)
        if free < cap + 1024**3:
            raise RuntimeError(f"CUDA memory headroom is insufficient: free={free}, requested cap={cap}")
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        report["gpu"] = {"device": str(device), "name": torch.cuda.get_device_name(device),
                         "initial_free_bytes": free, "allocation_cap_bytes": cap}
        from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
        from starVLA.model.framework.VLM4A.QwenGR00TRollingFlow import Qwen_GR00T_RollingFlow

        cfg, _ = read_mode_config(str(weights))
        cfg["trainer"]["pretrained_checkpoint"] = None
        seed_rng()
        model = Qwen_GR00T_RollingFlow(dict_to_namespace(cfg))
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model = model.to(device).eval()
        wrapper._framework = model
        report["checkpoint_load"] = {"status": "PASS", "strict": True, "class": type(model).__name__,
            "planning_horizon": model.action_horizon, "returned_horizon": model.execution_horizon}
        build = model.qwen_vl_interface.build_qwenvl_inputs
        action_predict = model.action_model.predict_action
        native_predict = model.predict_action

        def capture_inputs(*a, **kw):
            model_calls["vlm_input_build"] += 1
            result = build(*a, **kw)
            captured["arrays"] = {f"vlm/{key}": value.detach().cpu().numpy().copy() for key, value in result.items()}
            captured["dtypes"] = {key: str(value.dtype) for key, value in result.items()}
            return result

        def capture_state(features, state, **kw):
            model_calls["action_head_predict"] += 1
            captured["arrays"]["model/state"] = state.detach().float().cpu().numpy().copy()
            captured["state_dtype"] = str(state.dtype)
            return action_predict(features, state, **kw)

        def capture_prediction(*a, **kw):
            model_calls["framework_predict"] += 1
            result = native_predict(*a, **kw)
            captured["normalized_actions"] = np.array(result["normalized_actions"], copy=True)
            return result

        model.qwen_vl_interface.build_qwenvl_inputs = capture_inputs
        model.action_model.predict_action = capture_state
        model.predict_action = capture_prediction

    inverse_calls, state_calls = 0, 0
    state_apply, action_unapply = proc.apply_state, proc.unapply_actions

    def count_state(value):
        nonlocal state_calls
        state_calls += 1
        return state_apply(value)

    def count_inverse(value):
        nonlocal inverse_calls
        inverse_calls += 1
        return action_unapply(value)

    proc.apply_state, proc.unapply_actions = count_state, count_inverse
    for path in sorted(golden.glob("episode-*/training")):
        manifest, values = load_bundle(path)
        if manifest.provenance["checkpoint"]["sha256"] != before[str(weights)]:
            raise ValueError("golden samples were exported for different weights")
        raw, expected = arrays(manifest, values, "O1"), arrays(manifest, values, "O2")
        index = manifest.sample["loader_index"]
        if tuple(int(x) for x in dataset.all_steps[index]) != (manifest.sample["episode_id"], manifest.sample["frame_id"]):
            raise ValueError("current training split/sample identity differs from the golden")
        seed_rng()
        packed = dataset[index]
        cameras = manifest.layers[1].metadata["camera_order"]
        current = {f"image/{name}": np.asarray(image) for name, image in zip(cameras, packed["image"])}
        current.update(state=np.asarray(packed["state"]), action_labels=np.asarray(packed["action"]),
                       action_padding_mask=np.asarray(packed["action_padding_mask"]))
        stem = output / path.parent.name
        current_path = bundle(stem / "current-training", manifest, current, manifest.layers[1].metadata)
        loader_diff = compare_bundles(path, current_path, layers=["O2"])
        write_json(stem / "loader-diff.json", loader_diff)
        if loader_diff["status"] != "PASS":
            raise ValueError("current loader differs from the recorded training golden")
        adapter = Model.__new__(Model)
        adapter.model_cfg = {"image_resize_backend": "pillow_bicubic"}
        adapter.image_size, adapter.include_state = (224, 224), False
        observation = {"vision": {alias: {"color": raw[f"image/{name}"]} for alias, name in
            zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), cameras)}, "instruction": packed["lang"]}
        example = adapter._convert_obs(observation)
        # Explicit named raw flattening is only the model-side diagnostic input.
        # It does not pretend pack_robot_state maps a G2 head and waist.
        example["state"] = np.array(raw["state"], copy=True)
        state_before = example["state"].copy()
        normalized = wrapper._normalize_example_states([example], proc)[0]
        metadata = {"camera_order": cameras, "state_names": manifest.layers[0].metadata["state_names"],
                    "state_units": None, "boundary": "packed_images_normalized_state_before_VLM",
                    "instruction": packed["lang"]}
        expected_arrays = {key: value for key, value in current.items() if key == "state" or key.startswith("image/")}
        candidate = {f"image/{name}": image for name, image in zip(cameras, normalized["image"])}
        candidate["state"] = normalized["state"]
        reference = bundle(stem / "packed-reference", manifest, expected_arrays, metadata)
        observed = bundle(stem / "packed-candidate", manifest, candidate, metadata)
        diff = compare_bundles(reference, observed)
        write_json(stem / "packed-diff.json", diff)
        legacy = dict(candidate, state=legacy_proc.apply_state(raw["state"]).astype(np.float16))
        legacy_path = bundle(stem / "float32-counterexample", manifest, legacy, metadata, role="isolated_precision_counterexample")
        legacy_diff = compare_bundles(reference, legacy_path)
        write_json(stem / "float32-diff.json", legacy_diff)
        item = {"sample": manifest.sample, "loader_status": loader_diff["status"],
                "packed_status": diff["status"], "float32_counterexample_status": legacy_diff["status"],
                "raw_input_preserved": bool(np.array_equal(example["state"], state_before)),
                "model_tensor_status": "NOT_RUN", "prediction_status": "NOT_RUN"}
        report["samples"].append(item)
        if args.forward:
            model.reset(); seed_rng()
            target = model.predict_action(examples=[packed])["normalized_actions"]
            target_inputs, target_dtypes, target_state_dtype = captured["arrays"], captured["dtypes"], captured["state_dtype"]
            expected_env = []
            for batch in target:
                data, cursor = {}, 0
                for key in factory.action_keys:
                    dim = proc.action_key_dims[key]
                    data[key] = torch.as_tensor(np.array(batch[:, cursor:cursor + dim], copy=True))
                    cursor += dim
                result = dataset.transforms.unapply(data)
                expected_env.append(np.concatenate([np.asarray(result[key]) for key in factory.action_keys], axis=-1))
            before_calls = (state_calls, inverse_calls)
            before_model_calls = dict(model_calls)
            model.reset(); seed_rng()
            actual_env = wrapper.predict_action(examples=[example])["actions"]
            item["state_normalizations_per_prediction"] = state_calls - before_calls[0]
            item["action_inverses_per_prediction"] = inverse_calls - before_calls[1]
            item["model_calls_per_prediction"] = {key: value - before_model_calls[key] for key, value in model_calls.items()}
            model_meta = {**metadata, "boundary": "actual_VLM_inputs_and_action_head_state",
                          "vlm_dtypes": target_dtypes, "state_tensor_dtype": target_state_dtype,
                          "bfloat16_storage": "exact_values_stored_as_float32"}
            target_path = bundle(stem / "model-reference", manifest, target_inputs, model_meta)
            actual_meta = {**model_meta, "vlm_dtypes": captured["dtypes"], "state_tensor_dtype": captured["state_dtype"]}
            actual_path = bundle(stem / "model-candidate", manifest, captured["arrays"], actual_meta)
            tensor_diff = compare_bundles(target_path, actual_path)
            write_json(stem / "model-diff.json", tensor_diff)
            legacy_inputs = dict(target_inputs)
            legacy_state = torch.as_tensor(legacy_proc.apply_state(raw["state"])).to(
                getattr(torch, target_state_dtype.removeprefix("torch."))).float().numpy()[None]
            legacy_inputs["model/state"] = legacy_state
            legacy_path = bundle(stem / "model-float32-counterexample", manifest, legacy_inputs, model_meta, role="isolated_precision_counterexample")
            legacy_tensor_diff = compare_bundles(target_path, legacy_path)
            write_json(stem / "model-float32-diff.json", legacy_tensor_diff)
            # This checks inverse normalization against the genuine loader's
            # transform. Clipped/static/float16 labels cannot invert to raw labels.
            action_meta = {"action_names": manifest.layers[0].metadata["action_names"], "action_units": None,
                           "seed": seed, "model_reset_before_each_call": True,
                           "boundary": "checkpoint_output_and_training_transform_inverse"}
            target_path = bundle(stem / "action-reference", manifest, {"env_actions": np.stack(expected_env)}, action_meta, name="A0")
            actual_path = bundle(stem / "action-candidate", manifest, {"env_actions": actual_env}, action_meta, name="A0")
            exact_action_diff = compare_bundles(target_path, actual_path)
            write_json(stem / "action-exact-diff.json", exact_action_diff)
            # Loader inverse restores its recorded input dtype. The service
            # explicitly emits float32 numeric actions; retain the strict dtype
            # difference, then compare at that declared output boundary.
            output_meta = {**action_meta, "output_dtype": "float32",
                           "training_inverse_dtype": np.stack(expected_env).dtype.name}
            target_path = bundle(stem / "action-output-reference", manifest,
                {"normalized_actions": target, "env_actions": np.stack(expected_env).astype(np.float32)}, output_meta, name="A0")
            actual_path = bundle(stem / "action-output-candidate", manifest,
                {"normalized_actions": captured["normalized_actions"], "env_actions": actual_env}, output_meta, name="A0")
            action_diff = compare_bundles(target_path, actual_path)
            write_json(stem / "action-output-diff.json", action_diff)
            item.update(model_tensor_status=tensor_diff["status"], prediction_status=action_diff["status"],
                        inverse_exact_dtype_status=exact_action_diff["status"],
                        model_float32_counterexample_status=legacy_tensor_diff["status"],
                        returned_shape=list(actual_env.shape))
        write_json(output / "MODEL_BOUNDARY_REPORT.json", report)
    if not report["samples"]:
        raise ValueError("golden directory contains no training samples")
    after = snapshot(protected)
    report["provenance"]["source_file_sha256_after"] = after
    report["source_stable"] = before == after
    report["status"] = "PASS" if before == after and all(
        item["packed_status"] == "PASS" and item["raw_input_preserved"] and (not args.forward or
        (item["model_tensor_status"] == item["prediction_status"] == "PASS" and
         item["state_normalizations_per_prediction"] == item["action_inverses_per_prediction"] == 1))
        for item in report["samples"]) else "FAIL"
    if args.forward and any(item["model_calls_per_prediction"] != {key: 1 for key in model_calls} for item in report["samples"]):
        report["status"] = "FAIL"
    if args.forward:
        torch.cuda.synchronize(device)
        report["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "checkpoint", "golden", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--state-input-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--model-state-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-gib", type=float, default=8.)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or not np.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib <= 0:
        parser.error("thread count and GPU memory cap must be positive")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "G2_diagnostic_model_composition", "status": "RUNNING",
              "not_verified": ["full_registered_G2_adapter", "physical_units", "SDK_mapping", "EVA_wire", "task_success", "historical_training_source_revision"]}
    if not args.forward:
        report["not_verified"].extend(["final_model_tensors", "checkpoint_forward"])
    try:
        validate(args, report)
    except Exception as exc:
        report.update(status="FAIL", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if "provenance" in report:
            before = report["provenance"]["source_file_sha256"]
            after = snapshot([Path(path) for path in before])
            report["provenance"]["source_file_sha256_after"] = after
            report["source_stable"] = before == after
            if not report["source_stable"]:
                report["status"] = "FAIL"
        inventory = sorted(({"name": dist.metadata.get("Name"), "version": dist.version} for dist in importlib.metadata.distributions()), key=lambda item: str(item["name"]))
        write_json(args.output / "software-inventory.json", {"executable": sys.executable, "python": sys.version, "packages": inventory})
        report["software_inventory_sha256"] = file_sha256(args.output / "software-inventory.json")
        write_json(args.output / "MODEL_BOUNDARY_REPORT.json", report)
    print({"report": str(args.output / "MODEL_BOUNDARY_REPORT.json"), "status": report["status"]})
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
