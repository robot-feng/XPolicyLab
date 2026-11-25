"""RoboMME environment side of the XPolicyLab deployment loop."""
import argparse
from contextlib import ExitStack, closing
from importlib import import_module
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state


def observations(raw, goal):
    fields = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
    frames = []
    for front, wrist, joints, fingers in zip(*(raw[k] for k in fields), strict=True):
        cameras = {}
        for name, rgb in (("cam_head", front), ("cam_wrist", wrist)):
            rgb = np.asarray(rgb)
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid RGB frame for {name}: {rgb.shape} {rgb.dtype}")
            cameras[name] = {"color": rgb, "shape": rgb.shape[:2]}
        joints = np.asarray(joints, dtype=np.float32)
        fingers = np.asarray(fingers, dtype=np.float32)
        if joints.shape != (7,) or fingers.shape != (2,):
            raise ValueError(f"Invalid Panda joint/finger state: {joints.shape}, {fingers.shape}")
        frames.append({
            "vision": cameras, "state": {"arm_joint_state": joints, "ee_joint_state": fingers[:1]},
            "instruction": goal[0] if isinstance(goal, (list, tuple)) else goal,
            "env_idx": 0, "data_format_version": "v1.0",
        })
    if not frames:
        raise ValueError("RoboMME returned an empty observation")
    return frames


class Episode:
    """TASK_ENV for the existing single and batched deploy.py loops (one simulator)."""

    def __init__(self, env, dimensions, writer):
        self.env, self.dimensions, self.writer = env, dimensions, writer
        self.steps = 0
        self.terminated = self.truncated = False
        raw, self.info = env.reset()  # The builder owns the fixed episode seed.
        self.goal = self.info["task_goal"]
        *self.demonstration, self.obs = observations(raw, self.goal)
        self.record()

    def record(self):
        if self.writer is not None:
            frames = [self.obs["vision"][name]["color"] for name in ("cam_head", "cam_wrist")]
            self.writer.append_data(np.concatenate(frames, axis=1))

    def get_obs(self, env_idx=0):
        if env_idx != 0:
            raise IndexError(env_idx)
        return self.obs

    def get_running_env_idx_list(self):
        return [] if self.is_episode_end() else [0]

    def get_obs_batch(self, env_idx_list):
        return [self.get_obs(i) for i in env_idx_list]

    def is_episode_end(self):
        return self.terminated or self.truncated

    def take_action(self, action):
        vector = pack_robot_state({"state": action}, "joint", self.dimensions)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Expected a finite Panda joint action")
        raw, _, self.terminated, self.truncated, self.info = self.env.step(vector.astype(np.float32))
        self.obs = observations(raw, self.goal)[-1]
        self.steps += 1
        self.record()

    def take_action_batch(self, action_list, env_idx_list):
        if env_idx_list != [0] or len(action_list) != 1:
            raise ValueError("RoboMME runs one active environment per episode")
        self.take_action(action_list[0])


class EpisodeRPC:
    """Apply benchmark demo playback and execution horizon around the policy's RPC calls."""

    def __init__(self, client, episode, batch, horizon):
        self.client, self.episode, self.batch, self.horizon = client, episode, batch, horizon

    def call(self, func_name, obs=None):
        result = self.client.call(func_name=func_name, obs=obs)
        if func_name == "reset":
            update = "update_obs_batch" if self.batch else "update_obs"
            for frame in self.episode.demonstration:
                self.client.call(func_name=update, obs=[frame] if self.batch else frame)
        elif func_name in ("get_action", "get_action_batch"):
            chunks = result if func_name == "get_action_batch" else [result]
            if len(chunks) != 1 or not len(chunks[0]):
                raise ValueError("Expected one nonempty action chunk")
            chunk = chunks[0][:self.horizon]
            return [chunk] if func_name == "get_action_batch" else chunk
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_name", required=True)
    parser.add_argument("--task_name", dest="task", required=True, help="One native task name")
    parser.add_argument("--policy_name", dest="policy", required=True)
    parser.add_argument("--env_cfg_type", required=True)
    parser.add_argument("--additional_info", required=True)
    parser.add_argument("--root_dir", type=Path, required=True, help="Parent workspace")
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--seed", required=True, help="Policy run seed; the dataset fixes each episode seed")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--protocol", choices=("ws",), required=True)
    parser.add_argument("--eval_batch", choices=("true", "false"), required=True)
    parser.add_argument("--evaluation-id", default=os.getenv("ROBOMME_EVALUATION_ID"))
    parser.add_argument("--split", default=os.getenv("ROBOMME_SPLIT") or "test", choices=("train", "val", "test", "ood_test"))
    parser.add_argument("--start-episode", type=int, default=os.getenv("ROBOMME_START_EPISODE") or 0)
    parser.add_argument("--episodes", type=int, default=os.getenv("ROBOMME_EPISODES") or 1)
    parser.add_argument("--max-steps", type=int, default=os.getenv("ROBOMME_MAX_STEPS") or 1300)
    parser.add_argument("--action-steps", type=int, default=os.getenv("ROBOMME_ACTION_STEPS") or 16)
    parser.add_argument("--output", type=Path, default=os.getenv("ROBOMME_OUTPUT_DIR") or None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.env_cfg_type != "panda" or "action_type=joint" not in args.additional_info.split(","):
        parser.error("RoboMME requires panda and joint actions")
    if min(args.episodes, args.max_steps, args.action_steps) < 1 or args.start_episode < 0:
        parser.error("Episode count and step budgets must be positive, and start-episode nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    if args.task not in BenchmarkEnvBuilder.get_task_list(args.split):
        parser.error(f"Unknown task for {args.split}: {args.task}")
    batch = args.eval_batch == "true"
    deploy = import_module(f"XPolicyLab.policy.{args.policy}.deploy")
    evaluate = deploy.eval_one_episode_batch if batch else deploy.eval_one_episode
    dimensions = get_robot_action_dim_info("panda")
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.split, action_space="joint_angle",
                                  gui_render=False, max_steps=args.max_steps)
    end = args.start_episode + args.episodes
    if end > builder.get_episode_num():
        parser.error("Requested episode range exceeds the split")
    output = args.output or args.root_dir / "XPolicyLab/runs/robomme" / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)

    results = []
    evaluation_id = args.evaluation_id or output.name
    report = {"config": vars(args), "evaluation_id": evaluation_id, "episodes": results}
    try:
        for index in range(args.start_episode, end):
            trial = f"{args.task}-{args.split}-{index}"
            with ExitStack() as stack:
                client = stack.enter_context(WsModelClient(
                    url=f"ws://{args.host}:{args.port}", evaluation_id=evaluation_id,
                    trial_id=trial, request_timeout_s=600,
                ))
                env = stack.enter_context(closing(builder.make_env_for_episode(index)))
                writer = None
                if not args.no_video:
                    import imageio.v2 as imageio
                    writer = stack.enter_context(closing(imageio.get_writer(output / f"{trial}.mp4", fps=20)))
                episode = Episode(env, dimensions, writer)
                evaluate(episode, EpisodeRPC(client, episode, batch, args.action_steps))
                if not episode.is_episode_end():
                    raise RuntimeError("Deployment loop returned before native episode termination")
                seed, difficulty = builder.resolve_episode(index)
                result = {"episode": index, "benchmark_seed": seed, "difficulty": difficulty,
                          "status": episode.info["status"], "steps": episode.steps,
                          "initial_frames": len(episode.demonstration) + 1,
                          "terminated": bool(episode.terminated), "truncated": bool(episode.truncated)}
                results.append(result)
                print(f"{trial}: {result['status']} after {episode.steps} steps", flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=str(error), trial=trial)
        raise
    finally:
        report.update(completed=len(results), successes=sum(r["status"] == "success" for r in results))
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
