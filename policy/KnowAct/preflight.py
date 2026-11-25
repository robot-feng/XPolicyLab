"""Validate KnowAct's local source/assets before an expensive run.

This is a filesystem/environment preflight only. It does not load the model,
validate checkpoint keys, decode a dataset sample, or prove GPU kernel parity.
Those checks are separate acceptance gates described in docs/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch


POLICY_DIR = Path(__file__).resolve().parent
XPL_ROOT = POLICY_DIR.parents[1]
G05_ROOT = POLICY_DIR / "G05"


def _file_check(env_name: str) -> dict[str, Any]:
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser() if raw else None
    present = bool(path and path.is_file() and path.stat().st_size > 0)
    return {
        "path": str(path.resolve()) if path else None,
        "present": present,
        "kind": "non-empty file",
    }


def _directory_check(env_name: str, required_children: tuple[str, ...] = ()) -> dict[str, Any]:
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser() if raw else None
    exists = bool(path and path.is_dir())
    children = {
        name: bool(exists and (path / name).exists())
        for name in required_children
    }
    present = exists and (all(children.values()) if children else any(path.iterdir()))
    return {
        "path": str(path.resolve()) if path else None,
        "present": present,
        "kind": "directory",
        "required_children": children,
    }


def check() -> dict[str, Any]:
    python_version = tuple(sys.version_info[:3])
    python_compatible = (3, 10, 16) <= python_version < (3, 11, 0)

    assets = {
        "G05_PRETRAINED_CKPT": _file_check("G05_PRETRAINED_CKPT"),
        "G05_HF_PROCESSOR_PATH": _directory_check("G05_HF_PROCESSOR_PATH"),
        "G05_ACTION_TOKENIZER_CKPT": _file_check("G05_ACTION_TOKENIZER_CKPT"),
        "ROBODOJO_LEROBOT_V30_ROOT": _directory_check(
            "ROBODOJO_LEROBOT_V30_ROOT", ("meta", "data", "videos")
        ),
    }

    source = {
        "xpolicylab_g05_adapter": (XPL_ROOT / "policy/G05/model.py").is_file(),
        "xpolicylab_server": (XPL_ROOT / "setup_policy_server.py").is_file(),
        "knowact_adapter": (POLICY_DIR / "model.py").is_file(),
        "vendored_g05_policy": (G05_ROOT / "src/g05/models/g05/g05_policy.py").is_file(),
        "robodojo_task_config": (
            G05_ROOT / "configs/task/robodojo_arx_x5_joint.yaml"
        ).is_file(),
    }

    cuda = torch.cuda.is_available()
    all_assets_present = all(item["present"] for item in assets.values())
    all_source_present = all(source.values())

    return {
        "python": platform.python_version(),
        "python_compatible_with_vendored_g05": python_compatible,
        "torch": torch.__version__,
        "cuda_available": cuda,
        "gpu_count": torch.cuda.device_count(),
        "xpolicylab_root": str(XPL_ROOT),
        "knowact_root": str(POLICY_DIR),
        "source": source,
        "assets": assets,
        "ready_for_full_gpu_smoke_test": (
            python_compatible and cuda and all_assets_present and all_source_present
        ),
        "scope": (
            "Presence/environment check only. It does not load checkpoint weights, "
            "verify tokenizer compatibility, read a LeRobot sample, or run the simulator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless Python, CUDA, source and all required assets are present.",
    )
    parser.add_argument("--output", help="Optional path to write the JSON report.")
    args = parser.parse_args()

    result = check()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    if args.require_ready and not result["ready_for_full_gpu_smoke_test"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
