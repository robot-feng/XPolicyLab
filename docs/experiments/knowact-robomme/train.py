"""Native KnowAct CE on real, same-episode RoboMME windows under torchrun."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
from itertools import islice

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from initialize import REPO
from XPolicyLab.policy.KnowAct.demonstration import state_groups
from core import atomic_save, on_device
from data import Windows
from prepare import TASKS
from g05.data.native_episodes import (
    collate_episodes,
    executed_turn,
    history_indices,
    validate_history_processor,
    visual_turn,
)
from g05.utils.data.data_utils import collate_fn_pad_sequences
from g05.utils.checkpoint.ckpt_utils import find_run_dir
from scripts.serve_policy import setup


class Episodes(Windows):
    def __init__(self, root, split, processor, tasks=TASKS, demo_state_h5=None, state_frames=None):
        super().__init__(root, split, processor, tasks)
        limits = {r.get("demo_limit", 0) for _, r in self.rows}
        if len(limits) != 1:
            raise ValueError("Mixed demonstration sampling configurations")
        self.demo_limit = limits.pop()
        self.demo = {}
        self.demo_states = {}
        if self.demo_limit:
            for i, (path, r) in enumerate(self.rows):
                frames = np.load(path / "demo.npy", mmap_mode="r")
                if r.get("observation_timing") != "before_action" or len(frames) != min(self.demo_limit, r["demo_length"]):
                    raise ValueError(f"Incomplete or misaligned cache in {path}")
                self.demo[i] = frames

        if demo_state_h5 is not None:
            if not self.demo_limit or len(tasks) != 1:
                raise ValueError("One task with a prepared demonstration cache is required")
            with h5py.File(demo_state_h5, "r") as h5:
                for i, (_, r) in enumerate(self.rows):
                    ep = h5[f"episode_{r['episode']}"]
                    if ep["setup/seed"][()] != r["seed"] or r["demo_length"] != r["first_execution"]:
                        raise ValueError("Cached images and HDF5 demonstration prefix disagree")
                    groups = state_groups(r["demo_length"], self.demo_limit, state_frames or self.demo_limit)
                    if list(groups) != r["demo_indices"]:
                        raise ValueError("Cached RGB and state anchors disagree")
                    states = []
                    for indices in groups.values():
                        observations = [ep[f"timestep_{t}/obs"] for t in indices]
                        states.append({
                            "right_arm": torch.from_numpy(np.stack([o["joint_state"][()] for o in observations])).float(),
                            "right_gripper": torch.from_numpy(np.stack([o["gripper_state"][:1] for o in observations])).float(),
                        })
                    self.demo_states[i] = states

    def episode(self, index, turns):
        row, current = self.index[index]
        info = self.rows[row][1]
        stride = info["rgb_stride"]
        frames = history_indices(0, info["samples"], current, 16, turns)
        history = []
        for frame in frames:
            if (current - frame) % stride:
                raise ValueError(
                    "Execution history must align with a recorded RGB frame"
                )
            previous = index - (current - frame) // stride
            assert self.index[previous] == (row, frame)
            sample = self[previous]
            assert not sample["action_is_pad"][:16].any()
            history.append(executed_turn(sample, 16))
        return dict(self[index], native_history=history)


def full_pass_indices(dataset, world, rank, seed, offset):
    sampler = DistributedSampler(dataset, world, rank, seed=seed)
    epoch, skip = divmod(offset, len(sampler))
    while True:
        sampler.set_epoch(epoch)
        yield from islice(sampler, skip, None)
        epoch += 1
        skip = 0


def batch_from(dataset, rng, size, indices=None):
    items, rows = [], []
    for _ in range(size):
        index = next(indices) if indices is not None else int(rng.integers(len(dataset)))
        rows.append(dataset.index[index][0])
        items.append(dataset.episode(index, int(rng.integers(1, 5))))
    # native.train reads only the B=1 turns, not the legacy collated current batch.
    batch = collate_episodes(items, collate_fn_pad_sequences)
    for row, episode in zip(rows, batch["native_episodes"]):
        prefix = []
        for j, frame in enumerate(dataset.demo.get(row, ())):
            images = {key: torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
                      for key, rgb in zip(("image", "wrist_image"), frame)}
            state = dataset.demo_states[row][j] if row in dataset.demo_states else None
            prefix.append(visual_turn(dataset.proc, images, episode[-1]["samples"][0], state))
        episode[:0] = prefix
    return {"native_episodes": on_device(batch["native_episodes"])}


def select(model):
    model.requires_grad_(False)
    vlm = model.model.vlm
    if vlm.input_proj.weight is not vlm.output_proj.weight:
        raise RuntimeError("The pretrained LM head must remain tied")
    for part in (
        vlm.input_proj,
        vlm.layers[-4:],
        vlm.norm,
        model.model.proprio_embedder,
    ):
        part.float().requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def save_link(src, dest):
    tmp = dest.with_suffix(".link")
    tmp.unlink(missing_ok=True)
    os.link(src, tmp)
    os.replace(tmp, dest)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def main(a):
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    ini, run, data = a.initial.resolve(), a.output.resolve(), a.data.resolve()
    warm_start = a.warm_start.resolve() if a.warm_start else ini / "model_state_dict.pt"
    demo_state_h5 = a.demo_state_h5.resolve() if a.demo_state_h5 else None
    source = REPO / "policy/KnowAct/G05"
    os.chdir(source)
    seed_all(a.seed)
    cfg = OmegaConf.load(ini / ".hydra/config.yaml")
    checkpoint = run / "last.pt" if a.resume else warm_start
    vision = cfg.model.model_arch.vision
    if a.warm_start or a.resume:
        # A run's weights and its saved visual preprocessing are a pair.
        try:
            source_run = find_run_dir(str(checkpoint))
        except FileNotFoundError:
            source_run = ini  # Standalone weight files use the supplied initialization config.
        source_cfg = OmegaConf.load(source_run / ".hydra/config.yaml")
        vision.hf_image_patch_layout = source_cfg.model.model_arch.vision.get("hf_image_patch_layout", False)
    if a.hf_image_patches:
        vision.hf_image_patch_layout = True
    if not a.resume and (run / "last.pt").exists():
        raise FileExistsError("Existing training checkpoint; pass --resume")
    cfg.ckpt_path = str(checkpoint)
    cfg.run_dir = str(ini)
    model, procs = setup(cfg, device=f"cuda:{local}")
    params = select(model)
    kernel = next(
        m.chunk_gated_delta_rule
        for m in model.modules()
        if hasattr(m, "chunk_gated_delta_rule")
    )
    if not kernel.__module__.startswith("fla."):
        raise RuntimeError(f"Expected FLA, loaded {kernel.__module__}")
    if rank == 0:
        print(
            f"kernel={kernel.__module__} trainable={sum(p.numel() for p in params)} tensors={len(params)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(params, lr=a.lr, fused=True, weight_decay=0.01)
    plan = dict(
        seed=a.seed,
        steps=a.steps,
        batch=a.batch,
        world=world,
        lr=a.lr,
        initial=str(ini),
        data=str(data),
    )
    start = 0
    best = float("inf")
    if vision.get("hf_image_patch_layout", False):
        plan["hf_image_patch_layout"] = True
    if a.task or a.tasks or a.warm_start:
        plan.update(task=a.task, warm_start=str(warm_start))
    if demo_state_h5:
        plan["demo_state_h5"] = str(demo_state_h5)
    if a.demo_state_frames is not None:
        plan["demo_state_frames"] = a.demo_state_frames
    if a.tasks:
        plan["tasks"] = a.tasks
    if a.train_episode is not None:
        if len(a.train_episode) == 1:
            plan["train_episode"] = a.train_episode[0]  # Existing single-episode resume plans.
        else:
            plan["train_episodes"] = sorted(a.train_episode)
    if a.all_train_demos:
        plan["all_train_demos"] = True
    if a.resume or a.warm_start:
        saved = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
        if a.resume and saved["plan"] != plan:
            raise ValueError("Run configuration changed on resume")
        # setup casts for serving. Recover the exact master parameters for optimizer continuity.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(saved["model_state_dict"][name])
        if a.resume:
            start, best = saved["step"], saved["best"]
            optimizer.load_state_dict(saved["optimizer"])
        del saved
    net = DDP(
        model,
        device_ids=[local],
        gradient_as_bucket_view=True,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    train_proc = copy.deepcopy(procs["panda"])
    train_proc.train()
    validate_history_processor(train_proc)
    tasks = a.tasks or ((a.task,) if a.task else TASKS)
    train = Episodes(data, "all" if a.all_train_demos else "train", train_proc, tasks, demo_state_h5, a.demo_state_frames)
    val = Episodes(data, "all" if a.all_train_demos else "holdout", procs["panda"], tasks, demo_state_h5, a.demo_state_frames)
    metric_name = "train_probe_ce" if a.all_train_demos else "holdout_ce"
    indices = full_pass_indices(train, world, rank, a.seed, start * a.batch) if a.all_train_demos else None
    if a.train_episode is not None:
        selected = set(a.train_episode)
        train.index = [(row, t) for row, t in train.index if train.rows[row][1]["episode"] in selected]
        missing = selected - {train.rows[row][1]["episode"] for row, _ in train.index}
        if missing:
            raise ValueError(f"Episodes absent from the training split: {sorted(missing)}")
    if train.demo_limit != val.demo_limit:
        raise ValueError("Train and holdout demonstration sampling differ")
    if not train.demo_limit and not (a.legacy_execution_only or a.resume):
        raise ValueError("Prepare pre-action data with prepare_aligned.py or select --legacy-execution-only")
    cfg.model.model_arch.native_demo_frames = train.demo_limit
    cfg.model.model_arch.native_demo_proprio = demo_state_h5 is not None
    if a.demo_state_frames is not None:
        cfg.model.model_arch.native_demo_state_frames = a.demo_state_frames
    if rank == 0:
        (run / ".hydra").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints").mkdir(exist_ok=True)
        stats = run / "dataset_stats.json"
        if not stats.exists():
            stats.symlink_to(data / "run/dataset_stats.json")
        cfg.run_dir = str(run)
        OmegaConf.save(cfg, run / ".hydra/config.yaml", resolve=True)
        summary = dict(plan=plan, train=len(train), start=start)
        if a.all_train_demos:
            summary.update(probe_pool=len(val), probe_metric=metric_name)
        else:
            summary["validation"] = len(val)
        print(json.dumps(summary), flush=True)
    dist.barrier()

    @torch.no_grad()
    def validate():
        model.eval()
        results = torch.zeros(2, device="cuda")
        for i in range(rank, a.val_windows, world):
            seed_all(100000 + i)
            batch = batch_from(val, np.random.default_rng(100000 + i), 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(batch)
            results[0] += loss
            results[1] += 1
        dist.all_reduce(results)
        return float(results[0] / results[1])

    tick = time.monotonic()
    since = start
    stop = min(a.stop_at or a.steps, a.steps)
    if start >= stop:
        raise ValueError("Stop step must follow the current checkpoint")
    for step in range(start + 1, stop + 1):
        model.train()
        seed = a.seed + step * world + rank
        seed_all(seed)
        batch = batch_from(train, np.random.default_rng(seed), a.batch, indices)
        lr = (
            a.lr
            * min(1.0, step / 100)
            * (0.1 + 0.9 * (1 + np.cos(np.pi * step / a.steps)) / 2)
        )
        optimizer.param_groups[0]["lr"] = float(lr)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = net(batch)
        loss.backward()
        if step == start + 1 and any(p.grad is None for p in params):
            raise RuntimeError(
                "A statically selected parameter did not receive a gradient"
            )
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == start + 1 or step % 25 == 0:
            totals = loss.detach().float().clone()
            dist.all_reduce(totals)
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            step=step,
                            ce=float(totals) / world,
                            grad_norm=float(norm),
                            lr=float(lr),
                            windows_s=(step - since)
                            * world
                            * a.batch
                            / (time.monotonic() - tick),
                        )
                    ),
                    flush=True,
                )
            since = step
            tick = time.monotonic()
        if step % a.save_every == 0 or step in (10000, 20000) or step == stop:
            metric = validate()
            if not np.isfinite(metric):
                raise FloatingPointError(metric)
            if rank == 0:
                improved = metric < best
                best = min(metric, best)
                state = model.state_dict()
                record = {"step": step, "best": best, metric_name: metric, "plan": plan}
                milestone = step in (10000, 20000, 30000)
                # Keep a single resumable master. Serving snapshots omit Adam state.
                if improved or milestone:
                    path = (
                        run / "checkpoints" / f"step_{step:06d}.pt"
                        if milestone
                        else run / "best.pt"
                    )
                    atomic_save(dict(step=step, model_state_dict=state), path)
                    if improved and milestone:
                        save_link(path, run / "best.pt")
                atomic_save(
                    dict(
                        **record,
                        model_state_dict=state,
                        optimizer=optimizer.state_dict(),
                    ),
                    run / "last.pt",
                )
                with (run / "metrics.jsonl").open("a") as f:
                    print(json.dumps({"step": step, metric_name: metric}), file=f)
                print(metric_name if a.all_train_demos else "validation", step, metric, flush=True)
            dist.barrier()
            since = step
            tick = time.monotonic()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("initial", "output", "data"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--save-every", type=int, default=256)
    p.add_argument("--val-windows", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hf-image-patches", action="store_true", help="Use Qwen's official still-image patch layout when adapting older weights")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--task", help="One prepared official task")
    scope.add_argument("--tasks", nargs="+", help="Explicit list of prepared official tasks")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--train-episode", nargs="+", type=int, help="Fit the listed whole trajectories from the training split")
    selection.add_argument("--all-train-demos", action="store_true", help="Shuffle every official training window once per epoch, with DDP padding")
    p.add_argument("--warm-start", type=Path, help="Seed policy weights and start a fresh optimizer")
    p.add_argument("--demo-state-h5", type=Path, help="Pair the selected demonstration RGB frames with their observed joints from the official training HDF5")
    p.add_argument("--demo-state-frames", type=int, help="Number of demonstrated joint samples; default: match the stored RGB anchors")
    p.add_argument("--legacy-execution-only", action="store_true", help="Reproduce the original execution-only run")
    args = p.parse_args()
    if args.demo_state_frames is not None and (not args.demo_state_h5 or args.demo_state_frames < 1):
        p.error("--demo-state-frames requires --demo-state-h5 and a positive count")
    if args.tasks and len(set(args.tasks)) != len(args.tasks):
        p.error("--tasks contains a duplicate")
    if args.train_episode is not None and not args.task:
        p.error("--train-episode requires --task")
    if args.train_episode and len(set(args.train_episode)) != len(args.train_episode):
        p.error("--train-episode contains a duplicate")
    main(args)
