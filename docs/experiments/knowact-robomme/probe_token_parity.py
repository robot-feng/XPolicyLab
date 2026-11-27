"""Compare offline and streaming greedy decisions on the same expert prefixes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from g05.models.g05 import native_context
from probe_history import observation
from train import Episodes, REPO, batch_from
from XPolicyLab.policy.KnowAct.model import Model


class TwoTurns:
    """Select the current turn and, when available, its preceding turn in batch_from."""

    def integers(self, low, high):
        assert (low, high) == (1, 5)
        return 2


def offline_reference(policy, batch):
    ar = policy.model.ar_helper
    encoded = []
    reference = {}
    encode = native_context.encode
    loss_fn = ar.cal_ce_loss

    def trace_encode(*args, **kwargs):
        output = encode(*args, **kwargs)
        encoded.append((output[0], output[2]))
        return output

    def trace_loss(hidden, labels, model):
        sequence = torch.cat([ids for ids, _ in encoded], dim=1)[0]
        mask = encoded[-1][1][0]
        # Fractional 3.xx positions are the blocks in the final turn.
        positions = torch.where((mask.int() == 3) & (mask != 3))[0]
        positions += len(sequence) - len(mask)
        source = positions[: -ar.block_size]
        target = sequence[positions[ar.block_size :]]
        assert len(target) == ar._bar_expected_action_tokens
        assert torch.equal(labels[0, source + 1], target)
        end = torch.where(sequence[: positions[0]] == policy.processor.eov_token_id)[0][-1]
        reference.update(
            prefix=sequence[: end + 1].cpu(),
            target=target.cpu(),
            logits=model.vlm.decode(hidden[:, source]),
        )
        return loss_fn(hidden, labels, model)

    with (
        patch.object(native_context, "encode", trace_encode),
        patch.object(ar, "cal_ce_loss", trace_loss),
    ):
        native_context.train(policy, batch)
    return reference


def compare_decisions(model, reference, variable):
    ar = model.policy.model.ar_helper
    choose = ar._enforce_bar_joint_constraints
    offline, streaming = [], []

    def feed_teacher(logits, sampled, offset, history):
        block = slice(offset, offset + len(sampled))
        allowed = logits.isfinite()
        full = reference["logits"][:, block].masked_fill(~allowed, -torch.inf)
        offline.extend(choose(full, full[0].argmax(-1), offset, history).tolist())
        streaming.extend(choose(logits, sampled, offset, history).tolist())
        return reference["target"][block].to(sampled)

    with patch.object(ar, "_enforce_bar_joint_constraints", feed_teacher):
        model.get_action()
    state, _, _ = model._native_states[0]
    prefix = state.input_ids[0].cpu()
    off, live, truth = [
        torch.as_tensor(x)[variable] for x in (offline, streaming, reference["target"])
    ]
    return dict(
        identical_prefix=torch.equal(prefix, reference["prefix"]),
        prefix_tokens=len(prefix),
        variable=len(truth),
        agreement=int((off == live).sum()),
        offline_correct=int((off == truth).sum()),
        streaming_correct=int((live == truth).sum()),
    )


def probe(args):
    data, output, weights = (p.resolve() for p in (args.data, args.output, args.weights))
    cfg = yaml.safe_load((REPO / "policy/KnowAct/deploy.yml").read_text())
    cfg.update(
        env_cfg_type="panda",
        action_type="joint",
        bench_name="RoboMME",
        seed=7,
        ckpt_path=str(weights),
    )
    model = Model(cfg)
    policy = model.policy.eval()
    ar = policy.model.ar_helper
    ds = Episodes(data, "all", model.processor["panda"], ("PatternLock",))
    if (
        model.demo_head
        or policy.config.get("native_demo_proprio")
        or model.action_steps != 16
        or not model.keep_history
        or ar.do_sample
        or ds.demo_limit != policy.config.get("native_demo_frames")
    ):
        raise ValueError(
            "Use the RGB-only greedy baseline, matching demo cache, 16-step execution and history"
        )
    variable = torch.tensor(
        [
            "fixed_token" not in ar._bar_position_constraints[j]
            for j in range(ar._bar_expected_action_tokens)
        ]
    )
    indices = {key: i for i, key in enumerate(ds.index)}
    rows = []
    for row, (folder, meta) in enumerate(ds.rows):
        if meta["episode"] not in args.episodes:
            continue
        model.reset()
        q = ds.state[row]
        rgb = np.load(folder / "rgb.npy", mmap_mode="r")
        prompt = meta["deployment_prompt"]
        for i, images in zip(meta["demo_indices"], ds.demo[row]):
            # The RGB-only checkpoint discards proprioception on demonstration frames.
            obs = observation(images, q[0], prompt)
            obs["demonstration_frame"] = dict(index=i, length=meta["demo_length"])
            model.update_obs(obs)
        for t in (0, 16):
            idx = indices[row, t]
            ref = offline_reference(policy, batch_from(ds, TwoTurns(), 1, iter([idx])))
            obs = observation(rgb[t // meta["rgb_stride"]], q[t], prompt)
            if t:
                obs["__knowact_execution__"] = {"plan": 0, "steps": t}
            model.update_obs(obs)
            rows.append(
                dict(episode=meta["episode"], step=t, **compare_decisions(model, ref, variable))
            )
            if t == 0:
                # The next decision must see exactly the expert's executed history.
                state, plan, sent = model._native_states[0]
                pending = state.pending_action
                actual = ds[idx]["samples"]["action"]["value"].to(pending["action"]["value"])
                assert actual.shape == pending["action"]["value"].shape
                pending = {**pending, "action": {**pending["action"], "value": actual}}
                model._native_states[0] = replace(state, pending_action=pending), plan, sent
    if len(rows) != 2 * len(args.episodes):
        raise ValueError("Missing requested episode or anchor")
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "data", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe(parser.parse_args())
