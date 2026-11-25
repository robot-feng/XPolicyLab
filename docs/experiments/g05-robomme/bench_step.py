"""Compare training kernels on identical real observations, without updating a checkpoint."""
import argparse
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import configure
from core import on_device, select_parameters
from data import Windows
from g05.utils.data.data_utils import collate_fn_pad_sequences
from scripts.serve_policy import setup


def main(root, delta):
    root, delta = root.resolve(), delta.resolve()
    torch.manual_seed(7)
    cfg = configure(root, root / "run/action_tokenizer.pt")
    model, processors = setup(cfg, device="cuda:0")
    params = select_parameters(model)
    ckpt = torch.load(delta, weights_only=True, mmap=True, map_location="cpu")
    model.load_state_dict(ckpt["delta"], strict=False)
    opt = torch.optim.AdamW(params, lr=0., fused=True)
    model.train()
    proc = processors["panda"]
    proc.train()
    ds = Windows(root, "train", proc)
    ids = np.random.default_rng(55).integers(len(ds), size=(21, 2))
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    batches = iter(DataLoader(ds, batch_sampler=ids, collate_fn=collate_fn_pad_sequences, num_workers=0))
    fixed = next(batches)
    kernel = next(m.chunk_gated_delta_rule for m in model.modules() if hasattr(m, "chunk_gated_delta_rule"))
    rows = []
    for index in range(10):
        t = perf_counter()
        next(batches)  # measure the wait for a fresh input; replay fixed for numerical comparisons
        wait = perf_counter() - t
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(779)
        torch.cuda.synchronize()
        t = perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model(on_device(fixed))
        torch.cuda.synchronize()
        fwd = perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd = perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        end = perf_counter()
        if index >= 2:
            rows.append(dict(wait=wait, forward=fwd-t, backward=bwd-fwd, optimizer=end-bwd,
                             seconds=end-t+wait, ce=float(parts["ce_loss"]), fm=float(parts["fm_loss"]),
                             grad_norm=float(norm)))
    print(json.dumps(dict(step=ckpt["step"], kernel=kernel.__module__,
                          **{k: median(r[k] for r in rows) for k in rows[0]}), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("delta", type=Path, help="one saved core checkpoint, shared by both processes")
    args = p.parse_args()
    main(args.root, args.delta)
