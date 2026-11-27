"""Build a serving-compatible KnowAct initialization from the selected pretrained backbone."""

import argparse
import json
from pathlib import Path
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
SHARED = Path(__file__).resolve().parents[1] / "g05-robomme"
sys.path.insert(0, str(SHARED))
from config import configure


def configure_qwen(arch, snapshot):
    raw = json.loads((snapshot / "config.json").read_text())
    text, vision = raw["text_config"], raw["vision_config"]
    if raw["model_type"] != "qwen3_5" or (text["hidden_size"], vision["depth"]) != (
        1024,
        12,
    ):
        raise ValueError("Expected the Qwen3.5-0.8B multimodal architecture")
    for dst, src in ((arch.vlm, text), (arch.vision, vision)):
        for key in dst:
            if key in src:
                dst[key] = src[key]
    arch.vocab_size = arch.padded_vocab_size = text["vocab_size"]
    arch.hf_processor_path = arch.pretrained_model_path = str(snapshot)
    arch.vision.hf_image_patch_layout = True


def verify_qwen(model, snapshot):
    vlm, vision = model.model.vlm, model.model.vision_tower
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    language, visual = set(), set()
    for shard in sorted(set(index.values())):
        with safe_open(snapshot / shard, framework="pt") as weights:
            for key in weights.keys():
                if key.startswith("model.visual."):
                    name = key.removeprefix("model.visual.")
                    dst = vision.get_parameter(name)
                    visual.add(name)
                else:
                    prefix = next(
                        (
                            (old, new)
                            for old, new in vlm.DEFAULT_PREFIX_MAP
                            if key.startswith(old)
                        ),
                        None,
                    )
                    if prefix is None:
                        if key.startswith("mtp."):
                            continue
                        raise ValueError(f"Unrecognized Qwen tensor {key}")
                    name = prefix[1] + key[len(prefix[0]) :]
                    language.add(name)
                    dst = vlm.get_parameter(name)
                value = weights.get_tensor(key)
                if name == "input_proj.weight":
                    dst = dst[: len(value)]
                if not torch.equal(dst.detach().cpu(), value):
                    raise ValueError(f"Pretrained tensor differs: {key}")
    assert visual == set(dict(vision.named_parameters()))
    assert language == set(dict(vlm.named_parameters())) - {"output_proj.weight"}
    assert vlm.output_proj.weight is vlm.input_proj.weight
    return {"vision_tensors": len(visual), "language_tensors": len(language)}


def main(a):
    data, out = a.data.resolve(), a.output.resolve()
    source = REPO / "policy/KnowAct/G05"
    codec = data / "run/action_tokenizer.pt"
    snapshot = a.qwen_snapshot.resolve() if a.qwen_snapshot else None
    if (out / "model_state_dict.pt").exists():
        raise FileExistsError(out / "model_state_dict.pt")
    cfg = configure(data, codec, source=source)
    cfg.run_dir = cfg.output_dir = str(out)
    cfg.model.model_weights_to_bf16 = True
    arch = cfg.model.model_arch
    arch.native_context, arch.native_context_turns, arch.native_replan_steps = (
        True,
        4,
        16,
    )
    arch.discrete_action, arch.continuous_action, arch.return_continuous_action = (
        True,
        False,
        True,
    )
    arch.fm.joint_training, arch.fm.fm_weight = False, 0.0
    arch.ar.block_wise_autoregressive = True
    cfg.tokenizer.vq_config.block_wise_autoregressive = True
    arch.ar.block_size = cfg.tokenizer.vq_config.block_size = 16
    cfg.model.processor.discrete_action = True
    cfg.model.tokenizer = cfg.tokenizer
    torch.manual_seed(a.seed)
    if a.backbone == "qwen35-08b":
        if a.qwen_snapshot is None:
            raise ValueError(
                "--qwen-snapshot must point to an official local Qwen3.5-0.8B snapshot"
            )
        configure_qwen(arch, snapshot)
        model = instantiate(arch)
        report = verify_qwen(model, snapshot)
        # A zero state vector at the input of a pretrained RMSNorm stack gave
        # 1e12 proprio gradients on Panda. Start at a real word vector while
        # retaining the zero state-dependent projection and every Qwen weight.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        (state_id,) = tokenizer.encode("state", add_special_tokens=False)
        with torch.no_grad():
            model.model.proprio_embedder.mlp[-1].bias.copy_(
                model.model.vlm.input_proj.weight[state_id]
            )
        report["proprio_bias_token"] = state_id
    else:
        from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint

        model = load_model_from_checkpoint(arch, cfg.ckpt_path, device="cpu")
        report = {"source": cfg.ckpt_path}
    saved_codec = torch.load(codec, mmap=True, weights_only=True, map_location="cpu")
    expected = saved_codec["model_state_dict"]
    loaded = model.action_tokenizer.action_tokenizer.state_dict()
    if loaded.keys() != expected.keys() or any(
        not torch.equal(value.cpu(), expected[name]) for name, value in loaded.items()
    ):
        raise ValueError("Loaded ActionCodec differs from the first-stage checkpoint")
    report.update(codec_step=saved_codec["step"], codec_tensors=len(expected))
    assert (
        model.model.ar_helper.block_size == 16
        and model.model.ar_helper.block_wise_autoregressive
    )
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "model_state_dict.pt"
    arch.pretrained_model_path = None
    cfg.ckpt_path = str(dst)
    stats = out / "dataset_stats.json"
    if not stats.exists():
        stats.symlink_to(data / "run/dataset_stats.json")
    (out / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(cfg, out / ".hydra/config.yaml", resolve=True)
    tmp = dst.with_suffix(".tmp")
    torch.save({"model_state_dict": model.to(torch.bfloat16).state_dict()}, tmp)
    tmp.replace(dst)
    report.update(
        backbone=a.backbone,
        seed=a.seed,
        dataset=str(data),
        codec=str(codec),
        hidden_size=arch.vlm.hidden_size,
        vision_depth=arch.vision.depth,
        block_size=16,
        context_turns=4,
    )
    (out / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone", choices=("qwen35-08b", "g05"), required=True)
    p.add_argument("--qwen-snapshot", type=Path)
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
