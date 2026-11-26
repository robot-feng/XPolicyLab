"""Replan an expert continuation from a failed policy's prefix on a training seed."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from XPolicyLab.benchmarks.robomme.client import Episode
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state


def recover(trace, cutoff, output):
    record = json.loads(trace.read_text())
    if record["split"] != "train" or record["status"] not in ("fail", "timeout"):
        raise ValueError("Recovery labels require a failed training rollout")
    with np.load(trace.with_suffix(".npz")) as logged:
        if not 0 < cutoff < len(logged["q"]):
            raise ValueError("Select a decision point before termination")
        prefix, joints = logged["actions"][:cutoff], logged["q"][: cutoff + 1]
    builder = BenchmarkEnvBuilder(
        env_id="PatternLock",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    assert builder.resolve_episode(record["episode"])[0] == record["seed"]
    dims = get_robot_action_dim_info("panda")
    states, frames, actions = [], [], []
    with closing(builder.make_env_for_episode(record["episode"])) as env:
        episode = Episode(env, dims, None)
        for state, action in zip(joints, prefix):
            np.testing.assert_allclose(
                episode.obs["state"]["arm_joint_state"], state, rtol=0, atol=3e-4
            )
            episode.take_action(unpack_robot_state(action, "joint", dims))
        np.testing.assert_allclose(
            episode.obs["state"]["arm_joint_state"], joints[-1], rtol=0, atol=3e-4
        )
        world = env.unwrapped
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=world.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.3,
        )
        for _ in range(2 * len(world.selected_buttons)):
            if episode.is_episode_end():
                break
            visited = len(world.achieved_list)
            if world.achieved_list != world.selected_buttons[:visited] or visited == len(
                world.selected_buttons
            ):
                break
            # The native PatternLock expert aims the stick TCP 7 cm above the next button.
            xyz = world.selected_buttons[visited].pose.p.cpu().numpy()[0].copy()
            xyz[2] = 0.07
            target = sapien.Pose(xyz, world.agent.tcp.pose.q.cpu().numpy()[0])
            path = planner.move_to_pose_with_screw(target, dry_run=True)
            if not isinstance(path, dict):
                break
            for joint_target in path["position"]:
                obs = episode.obs
                if len(actions) % 4 == 0:
                    frames.append(
                        [obs["vision"][cam]["color"].copy() for cam in ("cam_head", "cam_wrist")]
                    )
                states.append(
                    np.concatenate([obs["state"][k] for k in ("arm_joint_state", "ee_joint_state")])
                )
                action = np.r_[joint_target, -1].astype(np.float32)
                actions.append(action)
                episode.take_action(unpack_robot_state(action, "joint", dims))
                if episode.is_episode_end():
                    break
        status = episode.info["status"] if episode.is_episode_end() else "incomplete"
        result = dict(
            episode=record["episode"],
            seed=record["seed"],
            cutoff=cutoff,
            end=episode.steps,
            status=status,
            target=record["target"],
            touched=[b.name for b in world.achieved_list],
        )
    output.mkdir(parents=True, exist_ok=False)
    if status == "success":
        np.savez_compressed(
            output / "correction.npz", rgb=frames, state=states, actions=actions, prefix=prefix
        )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace", type=Path, required=True, help="Training rollout JSON with matching NPZ"
    )
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    recover(a.trace, a.cutoff, a.out)
