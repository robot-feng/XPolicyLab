"""Trace native controls or replace one expert action chunk to localize a rollout failure."""

import argparse
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from client_server.ws import WsModelClient
from XPolicyLab.benchmarks.robomme.client import Episode, EpisodeRPC
from XPolicyLab.policy.KnowAct.deploy import eval_one_episode_batch
from XPolicyLab.policy.KnowAct.demonstration import frame_indices
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


class Trace(Episode):
    def __init__(self, *args, capture=False):
        super().__init__(*args)
        self.q, self.tcp, self.commands, self.contacts = [], [], [], []
        self.capture = capture
        self.rgb, self.state = [], []

    def take_action(self, action):
        if self.capture and self.steps % 16 == 0 and self.steps <= 48:
            self.rgb.append(
                [self.obs["vision"][k]["color"].copy() for k in ("cam_head", "cam_wrist")]
            )
            self.state.append(
                np.concatenate(
                    [self.obs["state"][k] for k in ("arm_joint_state", "ee_joint_state")]
                )
            )
        self.q.append(self.obs["state"]["arm_joint_state"].copy())
        self.tcp.append(
            self.env.unwrapped.agent.tcp.pose.p.detach().cpu().numpy().reshape(3).copy()
        )
        self.commands.append(pack_robot_state({"state": action}, "joint", self.dimensions).copy())
        super().take_action(action)
        buttons = [b.name for b in self.env.unwrapped.achieved_list]
        if buttons and (not self.contacts or self.contacts[-1]["buttons"] != buttons):
            self.contacts.append(dict(step=self.steps, buttons=buttons))


class RestartAtReplan(EpisodeRPC):
    """Re-send the same demonstration, discarding only the execution context."""

    def call(self, func_name, obs=None):
        if func_name == "update_obs_batch" and "__knowact_execution__" in obs[0]:
            super().call("reset")
            obs = [
                {k: v for k, v in frame.items() if k != "__knowact_execution__"} for frame in obs
            ]
        return super().call(func_name, obs)


def main(a):
    builder = BenchmarkEnvBuilder(
        env_id="PatternLock",
        dataset=a.split,
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    a.out.mkdir(parents=True, exist_ok=True)
    for i in a.episodes:
        seed, difficulty = builder.resolve_episode(i)
        expert = None
        if a.split == "train":
            folder = a.data / "data/PatternLock" / str(i)
            assert json.loads((folder / "episode.json").read_text())["seed"] == seed
            expert = np.load(folder / "action.npy")
        cases = dict.fromkeys(a.conditions or ["live"]) if a.port else {"expert": expert}
        if a.predictions:
            with np.load(a.predictions / f"predictions-{i}.npz") as archive:
                pred = archive["expert_history"]
                assert pred.shape == expert.shape
                cases = {"all_predicted_on_expert_obs": pred, "expert_codec": archive["codec"]}
                for t in range(0, len(expert), 16):
                    graft = expert.copy()
                    graft[t : t + 16] = pred[t : t + 16]
                    cases[f"replace_{t}"] = graft
        reset = None
        for case, sequence in cases.items():
            tag = f"{a.split}-{i}-{case}"
            with ExitStack() as stack:
                env = stack.enter_context(closing(builder.make_env_for_episode(i)))
                writer = (
                    stack.enter_context(closing(imageio.get_writer(a.out / f"{tag}.mp4", fps=20)))
                    if a.video
                    else None
                )
                ep = Trace(env, dims, writer, capture=bool(a.conditions) or a.capture_inputs)
                if a.port:
                    stride = int(case.split("_")[-1]) if a.conditions else 16
                    os.environ["KNOWACT_REPLAN_STEPS"] = str(stride)
                    if a.conditions:
                        if reset is None:
                            reset = ep.demonstration, ep.obs
                            selected = frame_indices(len(ep.demonstration), 4)
                            np.savez_compressed(
                                a.out / f"{a.split}-{i}-demo.npz",
                                indices=selected,
                                length=len(ep.demonstration),
                                rgb=[
                                    [
                                        ep.demonstration[j]["vision"][k]["color"]
                                        for k in ("cam_head", "cam_wrist")
                                    ]
                                    for j in selected
                                ],
                            )
                        # Fix the exact first policy inputs across interventions on this seed.
                        np.testing.assert_allclose(
                            ep.obs["state"]["arm_joint_state"],
                            reset[1]["state"]["arm_joint_state"],
                            atol=1e-6,
                            rtol=0,
                        )
                        ep.demonstration, ep.obs = reset
                    client = stack.enter_context(
                        WsModelClient(
                            url=f"ws://localhost:{a.port}",
                            evaluation_id=a.out.name,
                            trial_id=tag,
                            request_timeout_s=600,
                        )
                    )
                    rpc = RestartAtReplan if case.startswith("forget_") else EpisodeRPC
                    eval_one_episode_batch(ep, rpc(client, ep, True, stride))
                    assert ep.is_episode_end()
                else:
                    for action in sequence:
                        ep.take_action(unpack_robot_state(action, "joint", dims))
                        if ep.is_episode_end():
                            break
                status = ep.info["status"] if ep.is_episode_end() else "sequence_exhausted"
                result = dict(
                    episode=i,
                    split=a.split,
                    seed=seed,
                    difficulty=difficulty,
                    case=case,
                    steps=ep.steps,
                    status=status,
                    contacts=ep.contacts,
                    target=[b.name for b in env.unwrapped.selected_buttons],
                )
                np.savez_compressed(a.out / f"{tag}.npz", q=ep.q, tcp=ep.tcp, actions=ep.commands)
                if ep.capture:
                    np.savez_compressed(a.out / f"{tag}-inputs.npz", rgb=ep.rgb, state=ep.state)
                (a.out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
                print(tag, seed, status, ep.steps, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--port", type=int)
    group.add_argument("--predictions", type=Path)
    p.add_argument("--video", action="store_true")
    p.add_argument(
        "--capture-inputs",
        action="store_true",
        help="Save the first four 16-step observation anchors",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=("full_16", "repeat_16", "forget_16", "full_4", "full_8", "full_32"),
    )
    args = p.parse_args()
    if args.split == "val" and not args.port:
        p.error("Reference expert actions are available here only for train")
    if args.conditions and not args.port:
        p.error("Paired conditions require a policy server")
    main(args)
