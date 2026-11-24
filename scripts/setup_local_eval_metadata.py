"""Register minimal parent-workspace metadata for offline policy contract checks.

This does not install a simulator. Robot dimensions are copied from XPolicyLab's
training registry so the shared runtime and training helpers agree.
"""

import argparse
import json
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def register_metadata(workspace_root, env_cfg_type, num_envs=1):
    registry = json.loads((ROOT / "utils/robot/_robot_info.json").read_text())
    if env_cfg_type not in registry:
        raise ValueError(f"Register {env_cfg_type!r} in utils/robot/_robot_info.json first")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    config_root = Path(workspace_root) / "env_cfg"
    robot_path = config_root / "robot/_robot_info.json"
    env_path = config_root / f"{env_cfg_type}.yml"
    sim_path = config_root / "sim/xpolicylab_debug.yml"
    robot_info = json.loads(robot_path.read_text()) if robot_path.exists() else {}
    dimensions = registry[env_cfg_type]
    if env_cfg_type in robot_info and robot_info[env_cfg_type] != dimensions:
        raise ValueError(f"Existing robot metadata differs; leaving it untouched: {robot_path}")
    if env_path.exists():
        existing = yaml.safe_load(env_path.read_text())
        if existing.get("config", {}).get("robot") != env_cfg_type:
            raise ValueError(f"Existing environment names another robot: {env_path}")
    config_root.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.write_text(yaml.safe_dump({"config": {"robot": env_cfg_type, "sim": "xpolicylab_debug"}}))
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    if not sim_path.exists():
        sim_path.write_text(yaml.safe_dump({"scene": {"num_envs": num_envs}}))
    robot_info[env_cfg_type] = dimensions
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=robot_path.parent, delete=False) as stream:
        stream.write(json.dumps(robot_info, indent=2) + "\n")
        temporary = Path(stream.name)
    temporary.replace(robot_path)
    return {"env_cfg_type": env_cfg_type, "dimensions": dimensions,
            "environment": str(env_path), "robot_registry": str(robot_path),
            "scope": "minimal metadata; no simulator installed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_cfg_type")
    parser.add_argument("--workspace-root", type=Path, default=ROOT.parent)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(register_metadata(args.workspace_root, args.env_cfg_type, args.num_envs), indent=2))
