"""Episode splits, train-only statistics and samples shared by the two stages."""

import json
from pathlib import Path

import numpy as np
import torch

from prepare import TASKS


def episodes(root, split):
    rows = []
    for task in TASKS:
        paths = sorted((root / "data" / task).glob("*/episode.json"))
        records = [(p.parent, json.loads(p.read_text())) for p in paths]
        assert {r["episode"] for _, r in records} == set(range(100)), f"incomplete task: {task}"
        rows.extend(
            (p, r) for p, r in records if ("holdout" if r["episode"] % 11 == 0 else "train") == split
        )
    return rows


def statistics(root):
    train = episodes(root, "train")
    stats = {}
    for modality in ("action", "state"):
        x = np.concatenate([np.load(p / f"{modality}.npy") for p, _ in train])
        stats[modality] = {}
        for key, arr in [("right_arm", x[:, :-1]), ("right_gripper", x[:, -1:])]:
            low, high = np.quantile(arr, [0.01, 0.99], axis=0)
            stats[modality][key] = {
                f"global_{k}": v.tolist()
                for k, v in {
                    "q01": low,
                    "q99": high,
                    "min": arr.min(0),
                    "max": arr.max(0),
                    "mean": arr.mean(0),
                    "std": arr.std(0),
                }.items()
            }
    return {"panda": stats}


class Windows:
    def __init__(self, root, split, processor=None):
        self.rows = episodes(root, split)
        self.proc = processor
        self.act = [np.load(p / "action.npy") for p, _ in self.rows]
        self.state = [np.load(p / "state.npy") for p, _ in self.rows] if processor else None
        self.rgb = {}
        self.index = []
        for i, (_, r) in enumerate(self.rows):
            stride = r["rgb_stride"] if processor else 1
            stop = r["samples"] if processor else r["samples"] - 31
            self.index.extend((i, t) for t in range(0, stop, stride))
        stats = json.loads((root / "run/dataset_stats.json").read_text())["panda"]["action"]["right_arm"]
        self.lo, self.hi = (np.asarray(stats[f"global_{k}"], np.float32) for k in ("q01", "q99"))
        assert np.all(self.hi > self.lo)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, t = self.index[index]
        a = self.act[i]
        end = len(a)
        indices = t + np.arange(32)
        chunk = a[np.minimum(indices, end - 1)]
        if self.proc is None:
            return np.clip(2 * (chunk[:, :-1] - self.lo) / (self.hi - self.lo) - 1, -5, 5)
        p, r = self.rows[i]
        if i not in self.rgb:
            self.rgb[i] = np.load(p / "rgb.npy", mmap_mode="r")
        rgb = self.rgb[i][t // r["rgb_stride"]]
        state = torch.from_numpy(self.state[i][t : t + 1])
        action = torch.from_numpy(chunk)
        raw = dict(
            images={
                k: torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)
                for k, x in zip(("image", "wrist_image"), rgb)
            },
            state={"right_arm": state[:, :-1], "right_gripper": state[:, -1:]},
            action={"right_arm": action[:, :-1], "right_gripper": action[:, -1:]},
            action_is_pad=torch.from_numpy(indices >= end),
            state_is_pad=torch.zeros(1, dtype=torch.bool),
            image_is_pad=torch.zeros(1, dtype=torch.bool),
            task=r["prompt"],
            embodiment="panda",
            frequency=20.0,
            idx=index,
        )
        return self.proc.preprocess(raw)
