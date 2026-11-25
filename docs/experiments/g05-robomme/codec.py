"""Adapt the official ActionCodec on training trajectories and select on held-out trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from g05.tokenizer.models.actioncodec2_v2.modeling_actioncodec2v2 import ActionCodecV2Model

from data import Windows, statistics


def main(root, steps):
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    run = root / "run"
    run.mkdir(exist_ok=True)
    (run / "dataset_stats.json").write_text(json.dumps(statistics(root), indent=2) + "\n")
    train, val = (Windows(root, s) for s in ("train", "holdout"))
    batch = 24
    held = np.random.default_rng(42).choice(len(val), size=384, replace=False).reshape(-1, batch)
    model = ActionCodecV2Model.from_pretrained(root / "base/action_tokenizer_hf").cuda()
    model.rvq.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)

    def get(ds, ids):
        return torch.as_tensor(np.stack([ds[j] for j in ids]), device="cuda")

    @torch.no_grad()
    def error():
        model.eval()
        total = 0
        for ids in held:
            x = get(val, ids)
            codes = model.encode({"right_control": x})["right_control"][:, :2]
            recon = model.decode({"right_control": codes}, {"right_control": x.shape[-1]})[
                "right_control"
            ]
            total += (
                ((recon - x).abs() * torch.as_tensor((val.hi - val.lo) / 2, device="cuda")).mean().item()
            )
        return total / len(held)

    best = float("inf")
    metrics = []
    for step in range(steps + 1):
        if step % 128 == 0 or step == steps:
            mae = error()
            assert np.isfinite(mae)
            metrics.append(dict(step=step, holdout_arm_mae_rad=mae))
            print(metrics[-1], flush=True)
            if mae < best:
                best = mae
                tmp = run / "codec.tmp"
                torch.save(
                    {
                        "model_state_dict": {
                            "model." + k: v.cpu() for k, v in model.state_dict().items()
                        },
                        "tokenizer_meta": {
                            "parts_meta": {
                                "left_control": 9,
                                "left_gripper": 1,
                                "right_control": 9,
                                "right_gripper": 1,
                            }
                        },
                        "step": step,
                    },
                    tmp,
                )
                tmp.replace(run / "action_tokenizer.pt")
            (run / "codec_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if step == steps:
            break
        model.train()
        model.rvq.eval()  # keep pretrained code identities and EMA buffers fixed
        x = get(train, rng.integers(len(train), size=batch))
        opt.zero_grad(set_to_none=True)
        _, _, losses = model({"right_control": x}, d_original={"right_control": x.shape[-1]})
        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
    print("completed stage 1; selected MAE", best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--steps", default=1024, type=int)
    a = p.parse_args()
    main(a.root, a.steps)
