"""Teacher-forced CE at every cached camera anchor, with matched histories across checkpoints."""

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from train import Episodes, batch_from, seed_all, setup


@torch.inference_mode()
def evaluate(args):
    cfg = OmegaConf.load(args.weights.parent / ".hydra/config.yaml")
    cfg.run_dir, cfg.ckpt_path = str(args.weights.parent.resolve()), str(args.weights.resolve())
    model, processors = setup(cfg, device="cuda")
    if bool(model.config.get("native_demo_proprio", False)) != bool(args.demo_state_h5):
        raise ValueError("--demo-state-h5 must match the checkpoint demonstration modality")
    ds = Episodes(args.data, "all", processors["panda"], (args.task,), args.demo_state_h5,
                  model.config.get("native_demo_state_frames"))
    assert model.config.get("native_demo_frames") == ds.demo_limit
    with args.output.open("x", newline="") as target, torch.autocast("cuda", dtype=torch.bfloat16):
        writer = csv.writer(target)
        writer.writerow(["episode", "anchor", "ce", "accuracy"])
        for index in range(args.rank, len(ds), args.world):
            seed = 100000 + index
            seed_all(seed)
            row, t = ds.index[index]
            batch = batch_from(ds, np.random.default_rng(seed), 1, iter([index]))
            loss, _ = model(batch)
            writer.writerow(
                [ds.rows[row][1]["episode"], t, float(loss), model.train_action_accuracy]
            )
    print(f"Finished shard {args.rank} of {args.world} on {len(ds)} total anchors", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "data", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--demo-state-h5", type=Path, help="Official training source for demonstration-conditioned checkpoints")
    p.add_argument("--task", default="PatternLock")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.rank < a.world:
        p.error("Expected 0 <= rank < world")
    evaluate(a)
