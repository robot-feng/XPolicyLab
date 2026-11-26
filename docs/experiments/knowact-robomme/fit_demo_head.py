"""Fit a demonstration-conditioned action head from training rollouts of the frozen policy."""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from XPolicyLab.policy.KnowAct.demo_head import candidates, LEADS


def collect(a):
    import torch
    import yaml
    from probe_history import observation
    from train import REPO
    from XPolicyLab.policy.KnowAct.model import Model

    cfg = yaml.safe_load((REPO / "policy/KnowAct/deploy.yml").read_text())
    cfg.update(
        env_cfg_type="panda",
        action_type="joint",
        bench_name="RoboMME",
        seed=7,
        ckpt_path=str(a.weights.resolve()),
    )
    if os.environ.get("KNOWACT_DEMO_HEAD"):
        raise ValueError(
            "Collect from the unmodified baseline with KNOWACT_DEMO_HEAD unset"
        )
    a.h5, a.data, a.output = (p.resolve() for p in (a.h5, a.data, a.output))
    model = Model(cfg)
    if model.policy.config.get("native_demo_proprio", False):
        raise ValueError("This diagnostic was fitted against the RGB-only baseline")
    arm = "arm_joint_state"
    features, targets, episodes = [], [], []
    with h5py.File(a.h5, "r") as source, torch.inference_mode():
        for eid in range(a.rank, 100, a.world):
            folder = a.data / "data/PatternLock" / str(eid)
            meta = json.loads((folder / "episode.json").read_text())
            group = source[f"episode_{eid}"]
            assert int(group["setup/seed"][()]) == meta["seed"]
            demo = np.stack(
                [
                    group[f"timestep_{t}/obs/joint_state"][()]
                    for t in range(meta["first_execution"])
                ]
            )
            assert len(demo) == meta["demo_length"]
            q, rgb, action = [
                np.load(folder / f"{key}.npy", mmap_mode="r")
                for key in ("state", "rgb", "action")
            ]
            model.reset()
            for idx, image in zip(meta["demo_indices"], np.load(folder / "demo.npy")):
                demo_q = np.r_[demo[idx], group[f"timestep_{idx}/obs/gripper_state"][:1]]
                model.update_obs(
                    {
                        **observation(image, demo_q, meta["deployment_prompt"]),
                        "demonstration_frame": {"index": idx, "length": len(demo)},
                    }
                )
            phase = 0
            stride = model.action_steps
            for t in range(0, len(q), stride):
                obs = observation(
                    rgb[t // meta["rgb_stride"]], q[t], meta["deployment_prompt"]
                )
                if t:
                    obs["__knowact_execution__"] = {
                        "plan": t // stride - 1,
                        "steps": stride,
                    }
                model.update_obs(obs)
                base = np.stack([step[arm] for step in model.get_action()])[
                    : len(q) - t
                ]
                opts, phase = candidates(
                    demo, q[t, :-1], base, phase, stride if t else 0
                )
                features.append(opts - q[t, :-1, None])
                targets.append(action[t : t + len(base), :-1] - q[t, :-1])
                episodes.extend([eid] * len(base))
            print("collected", eid, flush=True)
    np.savez(
        a.output,
        x=np.concatenate(features),
        y=np.concatenate(targets),
        episode=episodes,
    )


def fit(a):
    from scipy.optimize import minimize

    shards = [np.load(p) for p in a.parts]
    x, y, ep = (np.concatenate([s[k] for s in shards]) for k in ("x", "y", "episode"))
    if set(ep) != set(range(100)) or sum(len(set(s["episode"])) for s in shards) != 100:
        raise ValueError(
            "Expected disjoint shards covering all 100 official training episodes"
        )
    train = ep % 11 != 0
    xt, yt = x[train].reshape(-1, x.shape[-1]).astype(float), y[train].ravel()
    gram = xt.T @ xt / len(yt)
    xy = xt.T @ yt / len(yt)
    # Convex weights keep the joint command within the observed candidates on each axis.
    n = x.shape[-1]
    res = minimize(
        lambda w: w @ gram @ w - 2 * w @ xy,
        np.full(n, 1 / n),
        jac=lambda w: 2 * (gram @ w - xy),
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(res.message)

    def metrics(sel):
        return dict(
            steps=int(sel.sum()),
            base_mae=float(np.abs(x[sel, :, 0] - y[sel]).mean()),
            fitted_mae=float(np.abs(x[sel] @ res.x - y[sel]).mean()),
        )

    artifact = dict(
        version=1,
        leads=LEADS,
        weights=res.x.tolist(),
        train=metrics(train),
        internal=metrics(~train),
    )
    a.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(artifact)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("collect")
    f = sub.add_parser("fit")
    for k in ("weights", "data", "h5"):
        c.add_argument("--" + k, type=Path, required=True)
    for k in ("rank", "world"):
        c.add_argument("--" + k, type=int, required=True)
    f.add_argument("--parts", nargs="+", type=Path, required=True)
    for cmd in (c, f):
        cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    (collect if args.mode == "collect" else fit)(args)
