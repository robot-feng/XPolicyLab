"""Paired train-only probe: swap current observations while preserving the live WS control path."""

import argparse
from contextlib import closing
import csv
import json
from pathlib import Path

import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from client_server.ws import WsModelClient
from XPolicyLab.benchmarks.robomme.client import EpisodeRPC
from XPolicyLab.policy.KnowAct.deploy import eval_one_episode_batch
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state
from probe_grasp import GraspTrace


class FeedbackTrace(GraspTrace):
    def __init__(self, env, dims, frames, states, stride, use_record):
        super().__init__(env, dims, None, "MoveCube")
        self.frames, self.states, self.stride = frames, states, stride
        self.use_record = use_record
        self.commands = []

    def is_episode_end(self):
        return super().is_episode_end() or self.steps >= len(self.states)

    def get_obs(self, env_idx=0):
        obs = super().get_obs(env_idx)
        if not self.use_record:
            return obs
        assert self.steps % self.stride == 0
        images = self.frames[self.steps // self.stride]
        q = self.states[self.steps]
        return {
            **obs,
            "vision": {
                k: {**obs["vision"][k], "color": v}
                for k, v in zip(("cam_head", "cam_wrist"), images, strict=True)
            },
            "state": {"arm_joint_state": q[:-1], "ee_joint_state": q[-1:]},
        }

    def take_action(self, action):
        self.commands.append(
            pack_robot_state({"state": action}, "joint", self.dimensions).copy()
        )
        super().take_action(action)


def main(a):
    b = BenchmarkEnvBuilder(
        env_id="MoveCube",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    a.out.mkdir(parents=True, exist_ok=True)
    for i in a.episodes:
        folder = a.data / "data/MoveCube" / str(i)
        meta = json.loads((folder / "episode.json").read_text())
        assert meta["seed"] == b.resolve_episode(i)[0]
        rgb = np.load(folder / "rgb.npy", mmap_mode="r")
        q = np.load(folder / "state.npy")
        runs = []
        for use_record in (False, True):
            with (
                closing(b.make_env_for_episode(i)) as env,
                WsModelClient(
                    url=f"ws://localhost:{a.port}",
                    evaluation_id=a.out.name,
                    trial_id=f"{i}-record-{int(use_record)}",
                    request_timeout_s=600,
                ) as client,
            ):
                ep = FeedbackTrace(env, dims, rgb, q, meta["rgb_stride"], use_record)
                assert ep.obs["instruction"] == meta["prompt"]
                eval_one_episode_batch(ep, EpisodeRPC(client, ep, True, 16))
                runs.append(np.stack(ep.commands))
                print(
                    i,
                    "recorded" if use_record else "live",
                    ep.steps,
                    ep.info["status"],
                    ep.events,
                    flush=True,
                )
        paired = min(map(len, runs))
        diff = np.abs(runs[0][:paired] - runs[1][:paired])
        with (a.out / f"feedback-{i}.csv").open("w") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(
                [
                    "episode",
                    "chunk_start",
                    "steps_compared",
                    "max_abs_arm_diff_rad",
                    "gripper_sign_mismatches",
                ]
            )
            for t in range(0, paired, 16):
                s = slice(t, min(t + 16, paired))
                w.writerow(
                    [
                        i,
                        t,
                        len(diff[s]),
                        float(diff[s, :-1].max()),
                        int(
                            np.count_nonzero(
                                (runs[0][s, -1] > 0) != (runs[1][s, -1] > 0)
                            )
                        ),
                    ]
                )
        np.savez(a.out / f"feedback-{i}.npz", live=runs[0], recorded=runs[1])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
