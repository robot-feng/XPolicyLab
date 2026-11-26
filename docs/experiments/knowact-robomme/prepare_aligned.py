"""Build pre-action windows using the simulator reset and recorded execution steps."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from XPolicyLab.benchmarks.robomme.client import observations
from XPolicyLab.policy.KnowAct.demonstration import frame_indices


def cameras(obs):
    return [obs["vision"][k]["color"] for k in ("cam_head", "cam_wrist")]


def prepare(args):
    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
        max_steps=1300,
    )
    directory = args.output / "data" / args.task
    directory.mkdir(parents=True, exist_ok=True)
    assets = args.output / "run"
    try:
        assets.symlink_to((args.source / "run").resolve(), target_is_directory=True)
    except FileExistsError:
        pass
    if not assets.samefile(args.source / "run"):
        raise ValueError("Output is linked to different normalization/codec assets")
    with h5py.File(args.h5 / f"record_dataset_{args.task}.h5", "r") as h5:
        for i in range(args.start, min(args.stop, builder.get_episode_num())):
            src = args.source / "data" / args.task / str(i)
            dest = directory / str(i)
            meta = json.loads((src / "episode.json").read_text())
            seed, _ = builder.resolve_episode(i)
            assert meta["seed"] == seed and meta["task"] == args.task
            info = dict(meta, observation_timing="before_action", demo_limit=args.demo_frames)
            if dest.exists():
                saved = json.loads((dest / "episode.json").read_text())
                if not all(saved.get(k) == v for k, v in info.items()):
                    raise ValueError(f"Incompatible existing cache at {dest}")
                continue
            with closing(builder.make_env_for_episode(i)) as env:
                raw, reset_info = env.reset()
                *demo, reset = observations(raw, reset_info["task_goal"])
            if (not demo and meta["first_execution"] > 0) or not reset["instruction"].strip():
                raise ValueError(f"Missing or mismatched reset for {src}")
            indices = frame_indices(len(demo), args.demo_frames)
            group = h5[f"episode_{i}"]
            assert int(group["setup/seed"][()]) == seed
            n, start, stride = meta["samples"], meta["first_execution"], meta["rgb_stride"]
            with TemporaryDirectory(dir=directory) as tmp:
                work = Path(tmp) / "episode"
                work.mkdir()
                selected = np.asarray([cameras(demo[j]) for j in indices], dtype=np.uint8)
                np.save(work / "demo.npy", selected.reshape(len(indices), *np.shape(cameras(reset))))
                after = np.load(src / "state.npy")
                first = np.concatenate(list(reset["state"].values()))
                np.save(work / "state.npy", np.vstack((first, after[:-1])).astype(after.dtype))
                shape = ((n + stride - 1) // stride, *np.shape(cameras(reset)))
                rgb = np.lib.format.open_memmap(
                    work / "rgb.npy", mode="w+", dtype="uint8", shape=shape
                )
                rgb[0] = cameras(reset)
                for t in range(stride, n, stride):
                    obs = group[f"timestep_{start + t - 1}/obs"]
                    rgb[t // stride] = [obs[k][()] for k in ("front_rgb", "wrist_rgb")]
                del rgb
                (work / "action.npy").symlink_to((src / "action.npy").resolve())
                (work / "episode.json").write_text(
                    json.dumps(
                        dict(
                            info,
                            deployment_prompt=reset["instruction"],
                            demo_length=len(demo),
                            demo_indices=indices,
                        )
                    )
                    + "\n"
                )
                work.rename(dest)
            print(f"{args.task} {i}: {n} actions, {len(demo)} demonstration frames", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "h5", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--task", default="PatternLock")
    p.add_argument("--demo-frames", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=100)
    args = p.parse_args()
    if args.demo_frames < 1 or not 0 <= args.start < args.stop:
        p.error("Expected a positive frame count and episode range")
    prepare(args)
