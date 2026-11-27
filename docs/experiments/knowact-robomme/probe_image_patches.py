"""Compare the exact inputs to KnowAct's frozen visual encoder with the checkpoint's HF processor."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.functional import cosine_similarity
from transformers import Qwen2VLImageProcessorFast

from g05.models.g05.g05_model_qwen35 import G05ModelQwen35
from g05.models.g05.qwen35.vision import Qwen3_5VisionModel
from g05.utils.checkpoint.ckpt_utils import find_run_dir


@torch.inference_mode()
def compare(args):
    cfg = OmegaConf.load(find_run_dir(str(args.weights)) / ".hydra/config.yaml").model.model_arch
    processor = Qwen2VLImageProcessorFast.from_pretrained(
        cfg.hf_processor_path, local_files_only=True
    )
    tower = Qwen3_5VisionModel(cfg.vision).eval()
    saved = torch.load(args.weights, weights_only=True, mmap=True, map_location="cpu")[
        "model_state_dict"
    ]
    prefix = "model.vision_tower."
    tower.load_state_dict({k[len(prefix) :]: v for k, v in saved.items() if k.startswith(prefix)})
    ctx = SimpleNamespace(cfg=cfg, vision_tower=tower, training=False)
    frames = np.load(args.images)
    # Use the actual four RGB demonstration frames and the two cameras at each time.
    rgb = torch.from_numpy(frames.copy()).permute(0, 1, 4, 2, 3)
    cams = rgb.transpose(0, 1).contiguous().flatten(0, 1)
    normalized = (cams.float() / 255 - 0.5) / 0.5
    inputs = {str(i): cam.unsqueeze(1) for i, cam in enumerate(normalized.chunk(2))}
    # Match resizing/normalization exactly before comparing the actual patch packing.
    hf = processor(
        images=list(normalized),
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        input_data_format="channels_first",
        return_tensors="pt",
    )
    patches = []
    hook = tower.register_forward_pre_hook(lambda _, args: patches.append(args[0]))
    stats, features = {}, []
    for mode in (False, True):
        cfg.vision.hf_image_patch_layout = mode
        features.append(G05ModelQwen35._forward_vision(ctx, inputs))
        delta = (patches[-1] - hf.pixel_values).abs()
        stats["hf" if mode else "legacy"] = dict(
            elements=delta.numel(),
            different=int(torch.count_nonzero(delta)),
            mean_abs=float(delta.mean()),
            max_abs=float(delta.max()),
        )
        assert patches[-1].shape == hf.pixel_values.shape
    hook.remove()
    stats["token_cosine_mean"] = float(cosine_similarity(*features, dim=-1).mean())
    assert stats["legacy"]["different"] > 0 and stats["hf"]["different"] == 0
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("images", "weights", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    torch.set_num_threads(4)
    compare(parser.parse_args())
