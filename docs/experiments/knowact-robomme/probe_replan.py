"""Swap just the image or proprioception at the second decision of a captured rollout."""

from pathlib import Path
import argparse
import json
import numpy as np
import yaml

from probe_history import observation
from train import REPO
from XPolicyLab.policy.KnowAct.model import Model
from XPolicyLab.utils.process_data import pack_robot_state


def main(a):
    folder = a.data / "data/PatternLock/1"
    prompt = json.loads((folder / "episode.json").read_text())["deployment_prompt"]
    truth = np.load(folder / "action.npy")
    live = np.load(a.live / "train-1-full_16-inputs.npz")
    captured = np.load(a.live / "train-1-full_16.npz")["actions"]
    expert = np.load(a.reference / "train-1-expert-inputs.npz")
    demo = np.load(a.live / "train-1-demo.npz")
    cfg = yaml.safe_load((REPO / "policy/KnowAct/deploy.yml").read_text())
    cfg.update(
        ckpt_path=str(a.weights.resolve()),
        bench_name="RoboMME",
        env_cfg_type="panda",
        action_type="joint",
        seed=7,
    )
    model = Model(cfg)
    first, result = None, {}

    def action(obs):
        model.update_obs(obs)
        return np.stack(
            [
                pack_robot_state({"state": v}, "joint", model.robot_action_dim_info)
                for v in model.get_action()
            ]
        )

    for images, image_data in (("expert", expert), ("live", live)):
        for state, state_data in (("expert", expert), ("live", live)):
            model.reset()
            for i, img in zip(demo["indices"], demo["rgb"]):
                model.update_obs(
                    {
                        **observation(img, live["state"][0], prompt),
                        "demonstration_frame": dict(index=int(i), length=int(demo["length"])),
                    }
                )
            initial = action(observation(live["rgb"][0], live["state"][0], prompt))
            if first is not None:
                np.testing.assert_allclose(initial, first, atol=1e-5, rtol=0)
            first = initial
            pred = action(
                {
                    **observation(image_data["rgb"][1], state_data["state"][1], prompt),
                    "__knowact_execution__": {"plan": 0, "steps": 16},
                }
            )
            result[f"image_{images}_state_{state}"] = float(
                np.abs(pred[:, :7] - truth[16:32, :7]).mean()
            )
            if images == state == "live":
                n = min(16, len(captured) - 16)
                result["live_second_reproduction_mae"] = float(
                    np.abs(pred[:n, :7] - captured[16 : 16 + n, :7]).mean()
                )
    result.update(
        live_first_reproduction_mae=float(np.abs(first[:, :7] - captured[:16, :7]).mean()),
        pre_second_state_difference_rad=float(
            np.abs(live["state"][1, :7] - expert["state"][1, :7]).mean()
        ),
        pre_second_pixel_mae=float(np.abs(live["rgb"][1].astype(float) - expert["rgb"][1]).mean()),
    )
    print(json.dumps(result), flush=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "live", "reference", "weights", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
