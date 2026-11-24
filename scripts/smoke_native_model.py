"""Exercise upstream inference before changing XPolicyLab or adding a simulator.

Run with the selected model's own Python. Synthetic RGB/state inputs establish
checkpoint loading and numerical inference only, never benchmark task success.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def rgb_image(height, width):
    import numpy as np
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :width // 3, 0] = 255
    image[:, width // 3:2 * width // 3, 1] = 255
    image[:, 2 * width // 3:, 2] = 255
    return image


def load_g05(args):
    import numpy as np
    import torch
    model_root = ROOT / "policy/G05/G05"
    # The upstream config loader eagerly resolves training-only environment
    # fields even for inference. No dataset is opened by serve_policy.setup.
    os.environ.setdefault("ROBODOJO_LEROBOT_V30_ROOT", "")
    sys.path[:0] = [str(model_root / "src"), str(model_root)]
    os.chdir(model_root)
    from g05.utils.config.config_resolvers import register_default_resolvers
    register_default_resolvers()
    from g05.models.g05.inferencer import PolicyInferencer, resolve_processor
    from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
    from g05.utils.eval.eval_utils import filter_embodiment
    from scripts.serve_policy import build_obs_dict, setup

    if not torch.cuda.is_available():
        raise RuntimeError("G05 CUDA is unavailable")
    torch.manual_seed(args.seed)
    overrides = [
        "eval_embodiment=robodojo", "model.model_arch.discrete_action=false",
        "model.model_arch.continuous_action=true",
        "model.model_arch.return_continuous_action=true",
        "model.processor.discrete_action=false", "model.use_torch_compile=false",
    ]
    cfg = load_config_from_run_dir(find_run_dir(str(args.checkpoint)), str(args.checkpoint), overrides)
    filter_embodiment(cfg, "robodojo")
    policy, processor = setup(cfg, device="cuda")
    inferencer = PolicyInferencer(policy, processor, device="cuda", action_source="fm")
    raw = {"images": {}, "state": {}, "task": "stack the bowls",
           "frequency": 30.0, "embodiment_type": "robodojo"}
    sub = resolve_processor(processor, raw)
    for entry in sub.shape_meta["images"]:
        shape = entry["raw_shape"]
        raw["images"][entry["key"]] = rgb_image(shape[-2], shape[-1]).transpose(2, 0, 1)
    for entry in sub.shape_meta["state"]:
        raw["state"][entry["key"]] = np.zeros(entry["raw_shape"], dtype=np.float32)

    def infer():
        with torch.inference_mode():
            result = inferencer.infer([build_obs_dict(raw, processor)])[0]
        torch.cuda.synchronize()
        result = {
            key: value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for key, value in result.items() if not key.startswith("_")
        }
        result = {key: value[0] if value.ndim == 3 and value.shape[0] == 1 else value
                  for key, value in result.items()}
        for entry in sub.shape_meta["action"]:
            value = np.asarray(result[entry["key"]])
            if value.ndim != 2 or value.shape[-1] != entry["raw_shape"]:
                raise ValueError(f"Unexpected G05 action layout for {entry['key']}: {value.shape}")
        return result

    return infer, {"backend": "upstream g05 PolicyInferencer", "embodiment": "robodojo",
                   "device": torch.cuda.get_device_name(0), "hydra_overrides": overrides,
                   "processor_path": os.environ.get("G05_HF_PROCESSOR_PATH"),
                   "training_dataset_used": False}


def load_pi05(args):
    import numpy as np
    import jax
    from openpi.models import pi0_config
    from openpi.policies import policy_config
    from openpi.training import config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Pi05 JAX GPU is unavailable")
    # Exact architecture/data settings of upstream's released pi05_libero config.
    # The vendored XPolicyLab training registry contains RoboDojo configs only.
    native_config = config.TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=config.LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
    )
    policy = policy_config.create_trained_policy(native_config, str(args.checkpoint))
    rng = np.random.default_rng(args.seed)
    observation = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": rgb_image(224, 224),
        "observation/wrist_image": rgb_image(224, 224),
        "prompt": "pick up the black bowl",
    }
    return lambda: {"actions": policy.infer(
        observation, noise=rng.normal(size=(10, native_config.model.action_dim)).astype(np.float32)
    )["actions"]}, {
        "backend": "upstream openpi create_trained_policy", "embodiment": "libero",
        "device": str(jax.devices()), "train_config": "pi05_libero",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["g05", "pi05"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "FAIL", "model": args.model, "checkpoint": str(args.checkpoint),
        "scope": "native inference on synthetic RGB/state; task success not measured",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "smoke_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {}, "iterations": [],
    }
    for package in ["torch", "torchvision", "jax", "jaxlib", "flax", "transformers", "numpy", "g05", "openpi"]:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.monotonic()
    try:
        import numpy as np
        np.random.seed(args.seed)
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        infer, details = (load_g05 if args.model == "g05" else load_pi05)(args)
        report.update(details)
        report["load_seconds"] = time.monotonic() - started
        for index in range(args.iterations):
            begin = time.monotonic()
            output = {key: np.asarray(value) for key, value in infer().items()}
            if not output:
                raise ValueError("The policy returned no action fields")
            for key, value in output.items():
                if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid action {key}: shape={value.shape}")
            if args.model == "pi05" and output["actions"].shape != (10, 7):
                raise ValueError(f"Unexpected pi05_libero output shape: {output['actions'].shape}")
            np.savez(args.output / f"actions_{index}.npz", **output)
            report["iterations"].append({
                "seconds": time.monotonic() - begin,
                "actions": {key: {"shape": list(value.shape), "min": float(value.min()),
                                  "max": float(value.max()), "finite": True}
                            for key, value in output.items()},
            })
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
