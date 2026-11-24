"""Evaluate joint-control policies through XPolicyLab RPC on native RoboMME episodes."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state


def observations(raw, instruction):
    """Preserve every reset/demo frame; Panda state uses the first physical finger."""
    keys = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
    lengths = [len(raw[key]) for key in keys]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"RoboMME observation histories must be nonempty and aligned: {lengths}")
    prompt = instruction[0] if isinstance(instruction, (list, tuple)) else instruction
    for front, wrist, joints, fingers in zip(*(raw[key] for key in keys), strict=True):
        joints = np.asarray(joints, dtype=np.float32).reshape(-1)
        fingers = np.asarray(fingers, dtype=np.float32).reshape(-1)
        if joints.shape != (7,) or fingers.shape != (2,):
            raise ValueError(f"Invalid Panda joints/fingers: {joints.shape}, {fingers.shape}")
        vision = {}
        for name, frame in (("cam_head", front), ("cam_wrist", wrist)):
            rgb = np.asarray(frame)
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Expected HWC uint8 RGB for {name}: {rgb.shape}, {rgb.dtype}")
            vision[name] = {"color": rgb, "shape": rgb.shape[:2]}
        yield {
            "vision": vision, "state": {"arm_joint_state": joints, "ee_joint_state": fingers[:1]},
            "instruction": prompt, "env_idx": 0, "data_format_version": "v1.0",
        }


def joint_action(action, dimensions):
    vector = np.asarray(pack_robot_state({"state": action}, "joint", dimensions, source_type="obs"), dtype=np.float32)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected finite Panda action (8,), got {vector}")
    return vector


def run_episode(env, client, dimensions, *, action_steps=16, eval_batch=True, writer=None):
    started = time.monotonic()
    raw, info = env.reset()  # The benchmark owns the fixed split/episode seed.
    client.call(func_name="reset")
    steps = chunks = frames = 0

    def update(raw_observation):
        nonlocal frames
        for obs in observations(raw_observation, info["task_goal"]):
            client.call(func_name="update_obs_batch" if eval_batch else "update_obs",
                        obs=[obs] if eval_batch else obs)
            frames += 1
        if writer is not None:
            writer.append_data(np.concatenate([raw_observation["front_rgb_list"][-1],
                                               raw_observation["wrist_rgb_list"][-1]], axis=1))

    update(raw)
    initial_frames = frames
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        actions = client.call(func_name="get_action_batch" if eval_batch else "get_action",
                              obs=[0] if eval_batch else None)
        if eval_batch:
            if len(actions) != 1:
                raise ValueError("Expected one action chunk for the single active RoboMME environment")
            actions = actions[0]
        if not len(actions):
            raise ValueError("Policy returned an empty action chunk")
        chunks += 1
        for action in actions[:action_steps]:
            raw, reward, terminated, truncated, info = env.step(joint_action(action, dimensions))
            steps += 1
            update(raw)
            if terminated or truncated:
                break
    return {
        "status": str(info.get("status", "unknown")), "success": info.get("status") == "success",
        "steps": steps, "chunks": chunks, "initial_frames": initial_frames, "observation_frames": frames,
        "reward": float(np.asarray(reward).item()), "terminated": bool(terminated), "truncated": bool(truncated),
        "task_goal": info["task_goal"], "duration_s": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Native task name, comma-separated names, or all")
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "ood_test"])
    parser.add_argument("--episodes", default="1", help="Number per task, or all")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--eval-batch", choices=["true", "false"], default="true")
    parser.add_argument("--policy-name", default="Pi_05")
    parser.add_argument("--checkpoint", help="Checkpoint identity recorded by the launcher")
    parser.add_argument("--seed", type=int, default=0, help="Policy seed recorded for provenance; never overrides episode seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.start_episode < 0 or args.max_steps < 1 or args.action_steps < 1:
        parser.error("Episode index must be nonnegative; step budgets must be positive")
    if args.episodes != "all" and (not args.episodes.isdigit() or int(args.episodes) < 1):
        parser.error("--episodes must be a positive integer or all")

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import robomme.env_record_wrapper as wrapper
    from client_server.ws import WsModelClient
    import imageio.v2 as imageio

    tasks = BenchmarkEnvBuilder.get_task_list(args.split) if args.task == "all" else args.task.split(",")
    dimensions = get_robot_action_dim_info("panda")
    output = args.output or ROOT / "runs/robomme" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "sample evaluation; not the full benchmark", "config": {**vars(args), "output": str(output)},
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_module": wrapper.__file__,
        "benchmark_source_sha256": {
            name: hashlib.sha256((Path(wrapper.__file__).parent / name).read_bytes()).hexdigest()
            for name in ("episode_config_resolver.py", "DemonstrationWrapper.py", "RecordWrapper.py")
        },
        "versions": {p: importlib.metadata.version(p) for p in ("robomme", "mani-skill", "sapien", "numpy")},
        "episodes": [],
    }
    client = None
    try:
        client = WsModelClient(url=f"ws://{args.host}:{args.port}", evaluation_id=output.name,
                               request_timeout_s=600, max_connect_attempts=3)
        for task in tasks:
            builder = BenchmarkEnvBuilder(env_id=task, dataset=args.split, action_space="joint_angle",
                                          gui_render=False, max_steps=args.max_steps)
            total = builder.get_episode_num()
            stop = total if args.episodes == "all" else args.start_episode + int(args.episodes)
            if not 0 <= args.start_episode < stop <= total:
                raise ValueError(f"Episode range [{args.start_episode}, {stop}) exceeds {task}/{args.split}: {total}")
            for episode in range(args.start_episode, stop):
                env = writer = None
                benchmark_seed, difficulty = builder.resolve_episode(episode)
                result = {"task": task, "split": args.split, "episode": episode,
                          "benchmark_seed": benchmark_seed, "difficulty": difficulty}
                try:
                    env = builder.make_env_for_episode(episode)
                    if not args.no_video:
                        video = f"{task}-{args.split}-{episode}.mp4"
                        writer = imageio.get_writer(output / video, fps=20, codec="libx264")
                        result["video"] = video
                    result.update(run_episode(env, client, dimensions, action_steps=args.action_steps,
                                              eval_batch=args.eval_batch == "true", writer=writer))
                except Exception as error:
                    result.update(status="error", success=False, error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    report["episodes"].append(result)
                    (output / f"{task}-{args.split}-{episode}.json").write_text(json.dumps(result, indent=2) + "\n")
                    if writer is not None:
                        writer.close()
                    if env is not None:
                        env.close()
                print(json.dumps(result), flush=True)
    finally:
        report["completed"] = sum(r.get("status") != "error" for r in report["episodes"])
        report["successes"] = sum(r["success"] for r in report["episodes"])
        report["success_rate"] = report["successes"] / len(report["episodes"]) if report["episodes"] else None
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        if client is not None:
            client.close()
    print(f"[ROBOMME] Results: {output}", flush=True)


if __name__ == "__main__":
    main()
