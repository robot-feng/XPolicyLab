"""Swap four demonstration RGB frames between two feasible paths from the same button."""

import argparse
from contextlib import closing
import itertools
import json
from pathlib import Path

import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from client_server.ws import WsModelClient
from XPolicyLab.benchmarks.robomme.client import Episode, EpisodeRPC
from XPolicyLab.policy.KnowAct.demonstration import frame_indices
from XPolicyLab.policy.KnowAct.deploy import eval_one_episode_batch
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from trace_patternlock import Trace


def capture(builder, index, dims):
    with closing(builder.make_env_for_episode(index)) as env:
        ep = Episode(env, dims, None)
        world = env.unwrapped
        return dict(
            demo=ep.demonstration,
            obs=ep.obs,
            target=[b.name for b in world.selected_buttons],
            positions={b.name: b.pose.p[0, :2].cpu().numpy().copy() for b in world.buttons_grid},
        )


class FirstTransition(Trace):
    def is_episode_end(self):
        # Diagnostic stop only. Neither target identity nor distances are sent to the policy.
        return (
            super().is_episode_end()
            or self.steps >= 96
            or (bool(self.contacts) and len(self.contacts[-1]["buttons"]) > 1)
        )


def run(builder, dims, args, recipient, donor, samples, condition):
    receiver = samples[recipient]
    source = samples[donor] if condition == "swap" else receiver
    first, expected = source["target"][:2]
    with closing(builder.make_env_for_episode(recipient)) as env:
        ep = FirstTransition(env, dims, None)
        assert [b.name for b in env.unwrapped.selected_buttons] == receiver["target"]
        for key in ep.obs["state"]:
            np.testing.assert_allclose(
                ep.obs["state"][key], receiver["obs"]["state"][key], atol=1e-6, rtol=0
            )
        # Hold the *exact* first policy input and the recipient's marker/time/state fields fixed.
        ep.obs = receiver["obs"]
        ep.demonstration = receiver["demo"].copy()
        slots = frame_indices(len(ep.demonstration), 4)
        for dest, src in zip(slots, frame_indices(len(source["demo"]), 4), strict=True):
            ep.demonstration[dest] = {
                **ep.demonstration[dest],
                "vision": source["demo"][src]["vision"],
            }

        label = f"val-{recipient}-alternative-{donor}-{condition}"
        with WsModelClient(
            url=f"ws://localhost:{args.port}",
            evaluation_id=args.out.name,
            trial_id=label,
            request_timeout_s=600,
        ) as ws:
            eval_one_episode_batch(ep, EpisodeRPC(ws, ep, True, 16))

        trace = np.vstack([*ep.tcp, env.unwrapped.agent.tcp.pose.p.cpu().numpy()[0]])
        touched = ep.contacts[-1]["buttons"] if ep.contacts else []
        direction = receiver["positions"][expected] - receiver["positions"][first]
        at = min(16, ep.steps)
        move = trace[at, :2] - trace[0, :2]
        cosine = (
            np.dot(move, direction) / (np.linalg.norm(move) * np.linalg.norm(direction))
            if np.linalg.norm(move) > 0.001
            else None
        )
        result = dict(
            recipient=recipient,
            alternative=donor,
            condition=condition,
            seed=builder.resolve_episode(recipient)[0],
            shown_path=source["target"],
            original_path=receiver["target"],
            touched=touched,
            steps=ep.steps,
            contacts=ep.contacts,
            matched_second=touched[:2] == [first, expected],
            initial_motion_mm=float(np.linalg.norm(move) * 1000),
            initial_motion_steps=at,
            initial_motion_cosine=float(cosine) if cosine is not None else None,
            closest_shown_second_mm=float(
                1000 * np.linalg.norm(trace[:, :2] - receiver["positions"][expected], axis=1).min()
            ),
        )
        np.savez_compressed(args.out / f"{label}.npz", commands=ep.commands, tcp=trace)
        (args.out / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(label, "shown", expected, "touched", touched, flush=True)


def main(args):
    if len(args.pairs) % 2:
        raise ValueError("Pass pairs of validation episode indices")
    args.out = args.out.resolve()
    args.out.mkdir(exist_ok=True, parents=True)
    builder = BenchmarkEnvBuilder(
        env_id="PatternLock",
        dataset="val",
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    for a, b in zip(args.pairs[::2], args.pairs[1::2]):
        samples = {i: capture(builder, i, dims) for i in (a, b)}
        x, y = samples[a], samples[b]
        assert x["target"][0] == y["target"][0] and x["target"][1] != y["target"][1]
        assert x["positions"].keys() == y["positions"].keys()
        for name in x["positions"]:
            np.testing.assert_allclose(
                x["positions"][name], y["positions"][name], rtol=0, atol=1e-6
            )
        for recipient, donor in itertools.permutations((a, b)):
            for condition in ("matched", "swap", "repeat"):
                run(builder, dims, args, recipient, donor, samples, condition)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pairs", type=int, nargs="+", required=True)
    main(parser.parse_args())
