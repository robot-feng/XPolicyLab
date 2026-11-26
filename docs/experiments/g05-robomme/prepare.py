"""Cache execution-only native RoboMME demonstrations; one directory per episode."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

TASKS = ("MoveCube", "PatternLock", "RouteStick", "InsertPeg")
RGB_STRIDE = 4  # action windows keep every control step


def convert(job):
    task, h5_root, metadata, out = job
    name = f"record_dataset_{task}"
    records = json.loads((metadata / "train" / f"{name}_metadata.json").read_text())["records"]
    assert len(records) == 100
    with h5py.File(h5_root / f"{name}.h5", "r") as source:
        for row in records:
            i = row["episode"]
            dest = out / task / str(i)
            if dest.exists():
                assert json.loads((dest / "episode.json").read_text())["seed"] == row["seed"]
                continue
            e = source[f"episode_{i}"]
            seed = int(e["setup/seed"][()])
            assert seed == row["seed"] and row["task"] == task
            end = sum(k.startswith("timestep_") for k in e)
            # The native schema has a demonstration prefix, then a contiguous execution phase.
            lo, hi = 0, end
            while lo < hi:
                mid = (lo + hi) // 2
                if e[f"timestep_{mid}/info/is_video_demo"][()]:
                    lo = mid + 1
                else:
                    hi = mid
            start = lo
            assert start < end
            n = end - start
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f".{i}-", dir=dest.parent))
            action = np.empty((n, *e[f"timestep_{start}/action/joint_action"].shape), np.float32)
            state = np.empty_like(action)
            images = np.lib.format.open_memmap(
                tmp / "rgb.npy",
                mode="w+",
                dtype=np.uint8,
                shape=((n + RGB_STRIDE - 1) // RGB_STRIDE, 2, 256, 256, 3),
            )
            for j in range(n):
                t = e[f"timestep_{start+j}"]
                assert not t["info/is_video_demo"][()]
                action[j] = t["action/joint_action"][()]
                obs = t["obs"]
                state[j] = np.r_[obs["joint_state"][()], obs["gripper_state"][:1]]
                if j % RGB_STRIDE == 0:
                    images[j // RGB_STRIDE] = [obs["front_rgb"][()], obs["wrist_rgb"][()]]
            del images
            assert np.isfinite(action).all() and np.isfinite(state).all()
            np.save(tmp / "action.npy", action)
            np.save(tmp / "state.npy", state)
            info = dict(
                row,
                samples=n,
                first_execution=start,
                rgb_stride=RGB_STRIDE,
                prompt=e["setup/task_goal"].asstr()[0],
            )
            (tmp / "episode.json").write_text(json.dumps(info) + "\n")
            tmp.rename(dest)
            print(task, i, n, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("h5_root", "metadata", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--tasks", nargs="+", default=TASKS, help="Official tasks to convert")
    a = p.parse_args()
    with ProcessPoolExecutor(min(4, len(a.tasks))) as pool:
        list(pool.map(convert, [(t, a.h5_root, a.metadata, a.out) for t in dict.fromkeys(a.tasks)]))
