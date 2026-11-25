"""Two-head Panda fine-tuning of the official G05 model. Run with torchrun."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from config import configure
from data import Windows
from g05.utils.data.data_utils import collate_fn_pad_sequences
from scripts.serve_policy import setup


def select_parameters(policy):
    policy.requires_grad_(False)
    selected = []
    for name, p in policy.named_parameters():
        k = name.split(".")
        if (
            (
                k[:2] in (["model", "vlm"], ["model", "action_expert"])
                and k[2] == "layers"
                and int(k[3]) >= 20
            )
            or (k[:2] == ["model", "action_expert"] and k[2] != "layers")
            or k[:2] == ["model", "proprio_embedder"]
        ):
            p.data = p.data.float()
            p.requires_grad_(True)
            selected.append(p)
    return selected


def on_device(item):
    if torch.is_tensor(item):
        return item.cuda(non_blocking=True)
    if isinstance(item, dict):
        return {k: on_device(v) for k, v in item.items()}
    if isinstance(item, list):
        return [on_device(v) for v in item]
    return item


def atomic_save(value, path):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


PEAK_LR, FLOOR_LR, WARMUP = 2e-5, 2e-6, 100
MILESTONES = (10_000, 20_000)


def lr_schedule(start, end, saved_lr):
    """Continue the cosine from its saved phase even when extending the total steps."""
    if not 0 <= start < end:
        raise ValueError(f"Expected 0 <= checkpoint step {start} < total steps {end}")
    phase = 0.0
    if start:
        envelope = saved_lr / min(1.0, start / WARMUP)
        cos = 2 * (envelope - FLOOR_LR) / (PEAK_LR - FLOOR_LR) - 1
        phase = math.acos(max(-1.0, min(cos, 1.0)))

    def lr(step):
        angle = phase + (math.pi - phase) * (step - start) / (end - start)
        return min(1.0, step / WARMUP) * (FLOOR_LR + (PEAK_LR - FLOOR_LR) * (1 + math.cos(angle)) / 2)

    return lr


def main(a):
    root = a.root.resolve()
    run = root / "run"
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    cfg = configure(root, run / "action_tokenizer.pt")
    model, processors = setup(cfg, device=f"cuda:{local}")
    params = select_parameters(model)
    opt = torch.optim.AdamW(params, lr=PEAK_LR, weight_decay=0.01, fused=True)
    net = DistributedDataParallel(
        model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True
    )
    proc = processors["panda"]
    proc.eval()
    ds = Windows(root, "holdout", proc)
    val_ids = np.random.default_rng(123).choice(len(ds), 128, replace=False)
    # Separate per-worker processors: validation must never inherit train augmentation.
    import copy

    train_proc = copy.deepcopy(proc)
    train_proc.train()
    train = Windows(root, "train", train_proc)
    # Keep the three original validation RNG streams and their batches identical across DDP sizes.
    validation = {
        shard: DataLoader(Subset(ds, val_ids[shard::3]), batch_size=2, collate_fn=collate_fn_pad_sequences)
        for shard in range(rank, 3, world)
    }
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    best = float("inf")
    start = 0
    if a.resume:
        s = torch.load(run / "resume.pt", weights_only=True, mmap=True, map_location="cpu")
        if s["delta"].keys() != names or not s["optimizer"]["state"]:
            raise ValueError("Checkpoint does not match the trainable parameters or lacks optimizer state")
        model.load_state_dict(s["delta"], strict=False)
        opt.load_state_dict(s["optimizer"])
        start = s["step"]
        best = s["best"]
        del s
    elif (run / "resume.pt").exists():
        raise FileExistsError("Run exists; use --resume")
    stop = a.stop_at or a.steps
    if not start < stop <= a.steps or (stop != a.steps and stop not in MILESTONES):
        raise ValueError("Stop at a future checkpoint milestone or the final step")
    schedule = lr_schedule(start, a.steps, opt.param_groups[0]["lr"])
    if rank == 0:
        if start:
            print(f"resuming optimizer at step {start}, saved_lr={opt.param_groups[0]['lr']:.10g}, "
                  f"next_lr={schedule(start + 1):.10g}", flush=True)
        (run / ".hydra").mkdir(exist_ok=True)
        OmegaConf.save(cfg, run / ".hydra/config.yaml")
        print(
            "trainable",
            sum(p.numel() for p in params),
            "windows",
            len(train),
            "holdout",
            len(ds),
            "updates",
            a.steps, "world_size", world, "global_batch", world * a.batch, "segment_end", stop,
            flush=True,
        )

    @torch.no_grad()
    def score():
        net.eval()
        acc = torch.zeros(2, device="cuda")
        with torch.random.fork_rng(devices=[local]):
            for shard, val in validation.items():
                torch.manual_seed(801 + shard)
                for batch in val:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = model(on_device(batch))
                    n = len(batch["idx"])
                    acc[0] += loss * n
                    acc[1] += n
        dist.all_reduce(acc)
        return (acc[0] / acc[1]).item()

    if a.resume:
        history = [json.loads(line) for line in (run / "core_metrics.jsonl").read_text().splitlines()]
        ref = next(entry["holdout_loss"] for entry in reversed(history) if entry["step"] == start)
        fresh = score()
        if rank == 0:
            print(f"resume validation {start}: reference={ref:.6f} current={fresh:.6f}", flush=True)
        if not math.isclose(ref, fresh, rel_tol=0.01, abs_tol=1e-5):
            raise RuntimeError("Resumed checkpoint failed fixed holdout comparison before optimization")

    rng = np.random.default_rng(7 + rank)
    # Per-step sampling is reproducible at an arbitrary resumed step.
    ids = rng.integers(len(train), size=(stop, a.batch))[start:]
    loader = DataLoader(
        train,
        batch_sampler=ids,
        collate_fn=collate_fn_pad_sequences,
        # Cached RGB windows are cheap to prepare; avoid forking a CUDA/Triton process.
        num_workers=0,
    )
    torch.manual_seed(7 + rank + start * world)
    tick = ready = time.perf_counter()
    first, input_wait = start, 0.0
    for step, batch in enumerate(loader, start + 1):
        input_wait += time.perf_counter() - ready
        net.train()
        lr = schedule(step)
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net(on_device(batch))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        opt.step()
        if step == start + 1 or step % 25 == 0:
            metrics = torch.stack([loss.detach(), parts["ce_loss"], parts["fm_loss"]]).float()
            dist.all_reduce(metrics)
            total, ce, fm = (metrics / world).tolist()
            if rank == 0:
                print(
                    json.dumps(dict(step=step, loss=total, ce=ce, fm=fm, grad_norm=norm.item(), lr=lr,
                                    samples_s=(step-first)*world*a.batch/(time.perf_counter()-tick),
                                    input_wait_ms=1000*input_wait/(step-first))),
                    flush=True,
                )
            tick, first, input_wait = time.perf_counter(), step, 0.0
        if step % 256 == 0 or step in MILESTONES or step == a.steps:
            metric = score()
            assert math.isfinite(metric)
            if rank == 0:
                delta = {k: v.detach().cpu() for k, v in model.named_parameters() if k in names}
                snapshot = {"step": step, "delta": delta, "holdout_loss": metric}
                if step in MILESTONES:
                    atomic_save(snapshot, run / f"step-{step}.pt")
                if metric < best:
                    atomic_save(snapshot, run / "best.pt")
                    best = metric
                atomic_save(
                    dict(step=step, delta=delta, optimizer=opt.state_dict(), best=best),
                    run / "resume.pt",
                )
                with (run / "core_metrics.jsonl").open("a") as f:
                    print(json.dumps(dict(step=step, holdout_loss=metric)), file=f)
                print("validation", step, metric, flush=True)
            dist.barrier()
            tick, first, input_wait = time.perf_counter(), step, 0.0
        ready = time.perf_counter()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=30_000, help="total optimizer updates, including resumed steps")
    p.add_argument("--stop-at", type=int, choices=MILESTONES, help="release GPUs after a milestone; keep the full LR schedule")
    p.add_argument("--resume", action="store_true")
    main(p.parse_args())
