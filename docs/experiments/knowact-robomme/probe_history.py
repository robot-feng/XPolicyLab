"""Compare history and within-chunk autoregression on identical expert observations."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import torch
import yaml

from train import Episodes, REPO
from XPolicyLab.policy.KnowAct.model import Model
from XPolicyLab.utils.process_data import pack_robot_state


def observation(images, state, prompt):
    return dict(
        vision={k: {"color": v} for k, v in zip(("cam_head", "cam_wrist"), images)},
        state={"arm_joint_state": state[:7], "ee_joint_state": state[7:]},
        instruction=prompt,
        env_idx=0,
    )


def probe(a):
    cfg = yaml.safe_load((REPO / "policy/KnowAct/deploy.yml").read_text())
    cfg.update(
        env_cfg_type="panda",
        action_type="joint",
        bench_name="RoboMME",
        seed=7,
        ckpt_path=str(a.weights.resolve()),
    )
    model = Model(cfg)
    ar = model.policy.model.ar_helper
    assert not ar.do_sample and ar.block_size == 16
    variable = np.array(
        [
            "fixed_token" not in ar._bar_position_constraints[j]
            for j in range(ar._bar_expected_action_tokens)
        ]
    )
    a.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "holdout"):
        ds = Episodes(a.data, split, model.processor["panda"], ("PatternLock",))
        indices = {key: i for i, key in enumerate(ds.index)}
        for row, (folder, meta) in enumerate(ds.rows):
            eid = meta["episode"]
            if eid not in a.episodes:
                continue
            rgb = np.load(folder / "rgb.npy", mmap_mode="r")
            predictions, metrics = {}, {}
            free_codes = []
            for mode in ("expert_history", "predicted_history", "no_history", "codec"):
                model.reset()
                model.keep_history = mode != "no_history"
                for frame_index, images in zip(meta["demo_indices"], ds.demo[row]):
                    model.update_obs(
                        {
                            **observation(images, ds.state[row][0], meta["deployment_prompt"]),
                            "demonstration_frame": dict(
                                index=frame_index, length=meta["demo_length"]
                            ),
                        }
                    )
                chunks, rows = [], []
                for t in range(0, len(ds.act[row]), 16):
                    truth = ds.act[row][t : t + 16]
                    sample = ds[indices[row, t]]["samples"]
                    ids, _, mask = model.policy.processor.preprocess(
                        samples=[sample],
                        device=model.policy.device,
                        training=False,
                        control_flag="return_suffix",
                    )
                    target = ids[(mask.int() == 3) & (mask != 3)].reshape(-1)[ar.block_size :]
                    assert len(target) == len(variable)
                    target_np = target.cpu().numpy()
                    observed = observation(
                        rgb[t // meta["rgb_stride"]], ds.state[row][t], meta["deployment_prompt"]
                    )
                    if t:
                        observed["__knowact_execution__"] = {"plan": t // 16 - 1, "steps": 16}
                    model.update_obs(observed)
                    teacher_choices = []
                    choose = ar._enforce_bar_joint_constraints

                    def feed_expert_codes(logits, sampled, offset, history):
                        chosen = choose(logits, sampled, offset, history)
                        teacher_choices.extend(chosen.tolist())
                        return target[offset : offset + len(sampled)].to(sampled)

                    override = patch.object(ar, "_enforce_bar_joint_constraints", feed_expert_codes)
                    with override if mode == "codec" else nullcontext():
                        chunk = np.stack(
                            [
                                pack_robot_state({"state": v}, "joint", model.robot_action_dim_info)
                                for v in model.get_action()
                            ]
                        )[: len(truth)]
                    state, plan, sent = model._native_states[0]
                    code = state.generated_ids[0, 1:-1].cpu().numpy()
                    err = dict(
                        step=t,
                        length=len(truth),
                        arm_mae=float(np.abs(chunk[:, :7] - truth[:, :7]).mean()),
                    )
                    if mode == "expert_history":
                        free_codes.append(code.copy())
                    if mode == "codec":
                        assert np.array_equal(code, target_np)
                        free = free_codes[t // 16]
                        teacher = np.array(teacher_choices)
                        assert np.array_equal(free[: ar.block_size], teacher[: ar.block_size])
                        for label, selector in (
                            ("first", np.arange(len(target)) < 16),
                            ("later", np.arange(len(target)) >= 16),
                        ):
                            sel = variable & selector
                            err[label] = {
                                "variable": int(sel.sum()),
                                "free_correct": int((free[sel] == target_np[sel]).sum()),
                                "teacher_correct": int((teacher[sel] == target_np[sel]).sum()),
                            }
                    if mode in ("expert_history", "codec"):
                        # After recording the prediction, acknowledge the actual expert commands.
                        # No target for this turn is present when its free prediction is made.
                        pending = state.pending_action
                        actual = sample["action"]["value"].to(pending["action"]["value"])
                        assert actual.shape == pending["action"]["value"].shape
                        pending = {**pending, "action": {**pending["action"], "value": actual}}
                        model._native_states[0] = replace(state, pending_action=pending), plan, sent
                    chunks.append(chunk)
                    rows.append(err)
                predictions[mode] = np.concatenate(chunks)
                metrics[mode] = rows
            for mode in ("predicted_history", "no_history"):
                np.testing.assert_allclose(
                    predictions[mode][:16], predictions["expert_history"][:16], atol=1e-5
                )
            np.savez(a.out / f"predictions-{eid}.npz", **predictions)
            (a.out / f"probe-{eid}.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(
                split,
                eid,
                {
                    k: round(float(np.abs(v[:, :7] - ds.act[row][:, :7]).mean()), 6)
                    for k, v in predictions.items()
                },
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "data", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe(p.parse_args())
