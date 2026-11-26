"""Capture a native expert rollout, then verify the training and serving inputs on it."""

import argparse
import json
from pathlib import Path

import numpy as np


def capture(a):
    from contextlib import closing
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    from XPolicyLab.benchmarks.robomme.client import Episode
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state

    builder = BenchmarkEnvBuilder(
        env_id="PatternLock",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    dims = get_robot_action_dim_info("panda")
    path = a.data / "data/PatternLock/0"
    assert builder.resolve_episode(0)[0] == json.loads((path / "episode.json").read_text())["seed"]

    def pixels(obs):
        return [obs["vision"][k]["color"] for k in ("cam_head", "cam_wrist")]

    with closing(builder.make_env_for_episode(0)) as env:
        ep = Episode(env, dims, None)
        frames = [pixels(o) for o in ep.demonstration]
        demo_state = [np.concatenate(list(o["state"].values())) for o in ep.demonstration]
        states, rgb = [], []
        for t, action in enumerate(np.load(path / "action.npy")):
            obs = ep.get_obs()
            states.append(np.concatenate(list(obs["state"].values())))
            if t % 4 == 0:
                rgb.append(pixels(obs))
            ep.take_action(unpack_robot_state(action, "joint", dims))
            if ep.is_episode_end():
                break
        assert ep.info["status"] == "success"
        np.savez_compressed(
            a.capture,
            demonstration=frames,
            demo_state=demo_state,
            state=states,
            rgb=rgb,
            instruction=ep.obs["instruction"],
        )


def check(a):
    import torch
    import yaml
    from unittest.mock import patch
    from train import Episodes, REPO, batch_from
    from XPolicyLab.policy.KnowAct.model import Model
    from g05.models.g05 import native_context

    root = a.data.resolve()
    demo_state_h5 = a.demo_state_h5.resolve() if a.demo_state_h5 else None
    data = np.load(a.capture)
    cfg = yaml.safe_load((REPO / "policy/KnowAct/deploy.yml").read_text())
    cfg.update(
        env_cfg_type="panda",
        action_type="joint",
        bench_name="RoboMME",
        seed=7,
        ckpt_path=str(a.weights.resolve()),
    )
    if a.demo_frames is not None:
        cfg["hydra_overrides"] = [f"model.model_arch.native_demo_frames={a.demo_frames}"]
    m = Model(cfg)
    ds = Episodes(root, "holdout", m.processor["panda"], ("PatternLock",), demo_state_h5,
                  m.policy.config.get("native_demo_state_frames"))
    with_proprio = bool(m.policy.config.get("native_demo_proprio", False))
    assert with_proprio == bool(a.demo_state_h5)
    assert m.policy.config.get("native_demo_frames") == ds.demo_limit
    row = next(i for i, (_, r) in enumerate(ds.rows) if r["episode"] == 0)
    folder, info = ds.rows[row]
    assert str(data["instruction"]) == info["deployment_prompt"]
    np.testing.assert_array_equal(ds.demo[row], data["demonstration"][info["demo_indices"]])
    np.testing.assert_allclose(ds.state[row][:20], data["state"][:20], atol=1e-5, rtol=0)
    cached = np.load(folder / "rgb.npy", mmap_mode="r")
    rgb_mae = float(np.abs(cached[:5].astype(float) - data["rgb"][:5]).mean())
    assert rgb_mae < 0.01, rgb_mae

    def obs(rgb, state):
        return dict(
            vision={k: {"color": x} for k, x in zip(("cam_head", "cam_wrist"), rgb)},
            state={"arm_joint_state": state[:7], "ee_joint_state": state[7:]},
            instruction=str(data["instruction"]),
            env_idx=0,
        )

    actions = np.load(folder / "action.npy")
    assert len(actions) == info["samples"]
    for t in range(0, len(actions), info["rgb_stride"]):
        ix = ds.index.index((row, t))
        indices = np.arange(t, t + 32)
        truth = torch.tensor(actions[np.minimum(indices, len(actions) - 1)])
        raw = m._build_obs_dict(
            m._to_g05_raw_obs(obs(cached[t // 4], ds.state[row][t])), m.processor
        )
        raw.update(
            action=dict(right_arm=truth[:, :7], right_gripper=truth[:, 7:]),
            action_is_pad=torch.from_numpy(indices >= len(actions)),
        )
        expected = ds.proc.preprocess(raw)
        actual = ds[ix]
        for k in ("proprio", "action", "action_is_pad"):
            assert torch.equal(expected[k], actual[k]), (t, k)
        assert all(
            torch.equal(expected["pixel_values"][k], v) for k, v in actual["pixel_values"].items()
        )
    t = 16
    item = ds.episode(ds.index.index((row, t)), 2)
    (prev,) = item["native_history"]
    assert torch.equal(prev["action"][:16], ds[ds.index.index((row, 0))]["action"][:16])
    assert prev["action_is_pad"][16:].all() and not prev["action_is_pad"][:16].any()
    assert torch.equal(prev["action"][16:], prev["action"][15:16].expand(16, -1))

    # Use the actual training batch builder, with the first control window selected.
    class FirstWindow:
        def integers(self, *args):
            return 1 if len(args) == 2 else ds.index.index((row, 0))

    if with_proprio:
        before = [{k: v.clone() for k, v in frame.items()} for frame in ds.demo_states[row]]
        batch_from(ds, FirstWindow(), 1)
        for raw, cached in zip(before, ds.demo_states[row]):
            for k, original in raw.items():
                torch.testing.assert_close(cached[k], original, rtol=0, atol=0)
    (episode,) = batch_from(ds, FirstWindow(), 1)["native_episodes"]
    demo_turns, current = episode[:-1], episode[-1]
    assert len(demo_turns) == len(info["demo_indices"]) and "executed_steps" not in current
    assert all(
        turn["supervise"] is False
        and "action" not in turn["samples"][0]
        and ("proprio" in turn["samples"][0]) == with_proprio
        for turn in demo_turns
    )

    calls = []
    append = native_context.append_observation

    def trace(policy, state, samples, pixels):
        i = len(calls)
        if i < len(demo_turns):
            train_turn = demo_turns[i]
            got, expected = samples[0], train_turn["samples"][0]
            assert got.keys() == expected.keys()
            for key in got:
                if key == "proprio":
                    for field in ("value", "proprio_dim_is_pad"):
                        torch.testing.assert_close(got[key][field].cpu(), expected[key][field].cpu(), rtol=0, atol=0)
                else:
                    assert got[key] == expected[key]
            assert all(
                torch.equal(p.cpu(), train_turn["pixel_values"][k].cpu()) for k, p in pixels.items()
            )
            full_ids, _, full_mask = policy.processor.preprocess(
                samples=samples, device=policy.device, training=True
            )
            prefix_ids, _, prefix_mask = policy.processor.preprocess(
                samples=samples, device=policy.device, training=False, control_flag="context_only"
            )
            assert torch.equal(full_ids[full_mask > 0], prefix_ids[prefix_mask > 0])
            assert torch.equal(full_mask[full_mask > 0], prefix_mask[prefix_mask > 0])
        calls.append(i)
        return append(policy, state, samples, pixels)

    m.reset()
    for i, (rgb, state) in enumerate(zip(data["demonstration"], data["demo_state"])):
        frame = obs(rgb, state)
        frame["demonstration_frame"] = dict(index=i, length=len(data["demonstration"]))
        m.update_obs_batch([frame])
    m.update_obs_batch([obs(data["rgb"][0], data["state"][0])])
    with patch.object(native_context, "append_observation", trace):
        (chunk,) = m.get_action_batch([0])
    assert len(calls) == len(demo_turns) + 1 and len(chunk) == 16
    m.reset()
    assert not m._demonstrations and not m._native_states
    print(
        json.dumps(
            dict(
                task="PatternLock",
                episode=0,
                seed=info["seed"],
                demo_proprio=with_proprio,
                demonstrated_state_tokens=sum(t["samples"][0]["proprio"]["value"].shape[0] for t in demo_turns) if with_proprio else 0,
                demonstration_indices=info["demo_indices"],
                native_context_appends=len(calls),
                action_window_anchors=len(range(0, len(actions), info["rgb_stride"])),
                output_steps=len(chunk),
                first_five_rgb_mae=rgb_mae,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "check"))
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--capture", required=True, type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--demo-state-h5", type=Path, help="Training source for checkpoints that use demonstrated joint observations")
    p.add_argument("--demo-frames", type=int, help="Override only when preflighting historical weights")
    args = p.parse_args()
    if args.mode == "check" and args.weights is None:
        p.error("--weights is required for the policy check")
    (capture if args.mode == "capture" else check)(args)
