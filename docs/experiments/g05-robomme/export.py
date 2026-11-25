"""Package the paired control / best validation checkpoint for the existing G05 server."""

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
from g05.tokenizer.models.actioncodec2_v2.modeling_actioncodec2v2 import ActionCodecV2Model

from config import configure


def export(root, kind, step=None):
    if step is not None and kind != "tuned":
        raise ValueError("--step selects a saved tuned checkpoint")
    folder = root / (f"eval-step-{step}" if step is not None else f"eval-{kind}")
    folder.mkdir(exist_ok=False)
    (folder / ".hydra").mkdir()
    codec = folder / "action_tokenizer.pt"
    base = root / "base/checkpoints/model_state_dict.pt"
    ckpt = folder / "model_state_dict.pt"
    if kind == "base":
        m = ActionCodecV2Model.from_pretrained(root / "base/action_tokenizer_hf")
        torch.save({"model_state_dict": {"model." + k: v for k, v in m.state_dict().items()}}, codec)
        os.link(base, ckpt)
    else:
        os.link(root / "run/action_tokenizer.pt", codec)
        source = "resume.pt" if kind == "last" else "best.pt"
        if step is not None:
            source = f"step-{step}.pt"
        best = torch.load(root / "run" / source, map_location="cpu", weights_only=True, mmap=True)
        assert step is None or best["step"] == step
        full = torch.load(base, map_location="cpu", mmap=True, weights_only=True)
        weights = full["model_state_dict"]
        assert best["delta"] and best["delta"].keys() <= weights.keys()
        for name, v in best["delta"].items():
            # The official 27D base loader takes the leading 20D grouped Panda projections.
            native = weights[name].shape
            if v.shape != native:
                assert name in {
                    "model.action_expert.input_proj.weight",
                    "model.action_expert.output_proj.weight",
                    "model.action_expert.output_proj.bias",
                    "model.proprio_embedder.mlp.0.weight",
                }, name
                assert tuple(27 if d == 20 else d for d in v.shape) == tuple(native), name
            assert v.isfinite().all(), name
            weights[name] = v
        torch.save(full, ckpt)
        print("exported core step", best["step"], flush=True)
    os.link(root / "run/dataset_stats.json", folder / "dataset_stats.json")
    cfg = configure(root, codec)
    OmegaConf.save(cfg, folder / ".hydra/config.yaml")
    print(ckpt, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("kind", choices=["base", "tuned", "last"])
    p.add_argument("--step", type=int, help="use one of the saved intermediate core steps")
    a = p.parse_args()
    export(a.root.resolve(), a.kind, a.step)
