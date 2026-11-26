"""Open-loop baseline: play only the public reset demonstration's measured joint positions."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from XPolicyLab.benchmarks.robomme.client import Episode
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from XPolicyLab.policy.KnowAct.demonstration import frame_indices


def evaluate(args):
    builder = BenchmarkEnvBuilder(
        env_id="PatternLock",
        dataset=args.split,
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    results = []
    for index in range(args.start, args.start + args.count):
        with closing(builder.make_env_for_episode(index)) as env:
            episode = Episode(env, dims, None)
            demo = [obs["state"]["arm_joint_state"] for obs in episode.demonstration]
            if not demo:
                raise ValueError("Expected a public demonstration")
            if args.demo_frames is not None:
                # Same observed frames as the policy; reconstruct only the public demonstration.
                keep = frame_indices(len(demo), args.demo_frames)
                values = np.asarray(demo)[keep].T
                demo = np.stack([np.interp(np.arange(len(demo)), keep, q) for q in values], axis=1).astype(np.float32)
            while not episode.is_episode_end():
                q = demo[min(episode.steps + 1, len(demo) - 1)]
                episode.take_action(
                    {"arm_joint_state": q, "ee_joint_state": np.array([-1], np.float32)}
                )
            # Native outcomes and the diagnostic labels are read after the last command.
            results.append(
                dict(
                    episode=index,
                    benchmark_seed=builder.resolve_episode(index)[0],
                    status=episode.info["status"],
                    steps=episode.steps,
                    target_buttons=[b.name for b in env.unwrapped.selected_buttons],
                    touched_buttons=[b.name for b in env.unwrapped.achieved_list],
                )
            )
            print(index, results[-1]["status"], flush=True)
    args.output.write_text(json.dumps({"demo_frames": args.demo_frames, "episodes": results}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--demo-frames", type=int, help="Uniform public keyframes; linearly interpolate in joint space (default: full trajectory)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.demo_frames is not None and args.demo_frames < 2:
        p.error("--demo-frames must be at least two")
    evaluate(args)
