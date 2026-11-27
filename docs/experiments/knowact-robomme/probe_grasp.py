"""Score KnowAct's grasp and placement separately in unmodified RoboMME rollouts."""

import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from client_server.ws import WsModelClient
from XPolicyLab.benchmarks.robomme.client import Episode, EpisodeRPC
from XPolicyLab.policy.KnowAct.deploy import eval_one_episode_batch
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state


class GraspTrace(Episode):
    def __init__(self, env, dims, writer, task):
        super().__init__(env, dims, writer)
        scene = env.unwrapped
        if task == "MoveCube":
            assert scene.way == "grasp_putdown", scene.way
            self.cube = scene.cube
        else:
            assert scene.num_repeats == 1 and len(scene.all_cubes) == 1
            self.cube = scene.target_cube
        self.start = scene.timestep
        assert not scene.task_list[self.start]["demonstration"]
        assert scene.task_list[self.start]["choice_label"].lower() == "pick up the cube"
        self.events = {}
        self.closest_m = float("inf")
        self.finger_range_m = [float("inf"), -float("inf")]

    def take_action(self, action):
        super().take_action(action)
        if self.info["status"] == "error":
            raise RuntimeError(
                self.info.get("error_message", "RoboMME returned an error")
            )
        scene = self.env.unwrapped
        tcp = scene.agent.tcp.pose.p
        self.closest_m = min(self.closest_m, (tcp - self.cube.pose.p).norm().item())
        finger = scene.agent.robot.get_qpos()[0, -2:].sum().item()
        self.finger_range_m = [
            min(self.finger_range_m[0], finger),
            max(self.finger_range_m[1], finger),
        ]
        for name, passed in [
            ("grasp", scene.agent.is_grasping(self.cube).item()),
            ("pickup", scene.timestep >= self.start + 1),
            ("placement", scene.timestep >= self.start + 2),
        ]:
            if passed:
                self.events.setdefault(name, self.steps)


def run(args):
    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset=args.split,
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    args.out.mkdir(parents=True, exist_ok=True)
    for i in args.episodes:
        seed, difficulty = builder.resolve_episode(i)
        assert difficulty == "easy"
        tag = f"{args.task}-{args.split}-{i}"
        if not args.port:
            tag += f'-{args.replay_kind or "expert"}-control'
        with ExitStack() as stack:
            env = stack.enter_context(closing(builder.make_env_for_episode(i)))
            writer = (
                stack.enter_context(
                    closing(imageio.get_writer(args.out / f"{tag}.mp4", fps=20))
                )
                if args.video
                else None
            )
            episode = GraspTrace(env, dims, writer, args.task)
            if not args.port:
                if args.expert_data:
                    folder = args.expert_data / "data" / args.task / str(i)
                    recorded_seed = json.loads((folder / "episode.json").read_text())[
                        "seed"
                    ]
                    commands = np.load(folder / "action.npy")
                else:
                    with np.load(args.predictions / f"predictions-{i}.npz") as archive:
                        recorded_seed = int(archive["seed"])
                        commands = archive[args.replay_kind]
                if recorded_seed != seed:
                    raise ValueError("Control actions do not match the native seed")
                # Hold the final absolute joint target until the same native deadline.
                while not episode.is_episode_end():
                    action = commands[min(episode.steps, len(commands) - 1)]
                    episode.take_action(unpack_robot_state(action, "joint", dims))
            else:
                client = stack.enter_context(
                    WsModelClient(
                        url=f"ws://localhost:{args.port}",
                        evaluation_id=args.out.name,
                        trial_id=tag,
                        request_timeout_s=600,
                    )
                )
                eval_one_episode_batch(episode, EpisodeRPC(client, episode, True, 16))
            if not episode.is_episode_end():
                raise RuntimeError("Rollout ended without native termination")
            result = dict(
                task=args.task,
                split=args.split,
                episode=i,
                seed=seed,
                difficulty=difficulty,
                instruction=episode.obs["instruction"],
                demonstration_frames=len(episode.demonstration),
                steps=episode.steps,
                status=episode.info["status"],
                first_step=episode.events,
                closest_tcp_to_cube_m=episode.closest_m,
                finger_opening_range_m=episode.finger_range_m,
            )
        (args.out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("PickXtimes", "MoveCube"), required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", type=int, help="KnowAct websocket port")
    source.add_argument(
        "--expert-data",
        type=Path,
        help="Matched training actions, for the separate control only",
    )
    source.add_argument(
        "--predictions", type=Path, help="Seed-tagged archives from probe_history.py"
    )
    p.add_argument(
        "--replay-kind", choices=("codec", "expert_history", "predicted_history")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    if bool(args.predictions) != bool(args.replay_kind):
        p.error("--replay-kind requires --predictions and vice versa")
    if not args.port and args.split != "train":
        p.error("Expert controls are restricted to the training split")
    run(args)
