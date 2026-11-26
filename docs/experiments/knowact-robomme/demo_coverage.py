"""Measure demonstration sampling coverage using annotations only for offline analysis."""

import argparse
from collections import Counter
from itertools import groupby
from pathlib import Path
import json

import h5py

from XPolicyLab.policy.KnowAct.demonstration import frame_indices


def coverage(path, source):
    rows = []
    with h5py.File(source, "r") as h5:
        for meta_path in sorted((path / "data/PatternLock").glob("*/episode.json")):
            m = json.loads(meta_path.read_text())
            ep = h5[f"episode_{m['episode']}"]
            length = m["demo_length"]
            assert int(ep["setup/seed"][()]) == m["seed"] and m["first_execution"] == length
            labels = [ep[f"timestep_{t}/info/simple_subgoal"].asstr()[()] for t in range(length)]
            if not all(s.startswith("move ") for s in labels):
                raise ValueError(f"Unexpected direction label in episode {m['episode']}")
            segment, path_labels = [], []
            for i, (name, frames) in enumerate(groupby(labels)):
                path_labels.append(name)
                segment.extend([i] * sum(1 for _ in frames))
            assert m["demo_indices"] == frame_indices(length, 4)
            hit = {k: len({segment[i] for i in frame_indices(length, k)}) for k in (4, 8, 16)}
            rows.append((m["episode"], tuple(path_labels), hit))
    assert len(rows) == 100
    train = [r for r in rows if r[0] % 11]
    held = [r for r in rows if not r[0] % 11]
    seen = {r[1] for r in train}
    return dict(
        train_episodes=len(train),
        holdout_episodes=len(held),
        unique_train_paths=len(seen),
        unseen_holdout_paths=sum(r[1] not in seen for r in held),
        train_first_direction_counts=Counter(r[1][0] for r in train),
        holdout_first_directions_seen=sum(any(s[1][0] == r[1][0] for s in train) for r in held),
        splits={
            name: {
                "direction_runs": sum(len(p) for _, p, _ in subset),
                "fully_covered": {k: sum(h[k] == len(p) for _, p, h in subset) for k in (4, 8, 16)},
                "covered_runs": {k: sum(h[k] for _, p, h in subset) for k in (4, 8, 16)},
            }
            for name, subset in (("train", train), ("holdout", held))
        },
        train1_sequence=next(r[1] for r in train if r[0] == 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "h5", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    result = coverage(args.data, args.h5)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
