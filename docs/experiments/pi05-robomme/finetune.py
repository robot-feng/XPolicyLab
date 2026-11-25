"""Short action-expert fine-tune using RoboMME's pinned training implementation."""
import argparse
from dataclasses import replace
import logging
from pathlib import Path

from flax import nnx
from openpi.shared.nnx_utils import PathRegex
from openpi.training.weight_loaders import CheckpointWeightLoader
from mme_vla_suite.training.config import get_config
import train  # robomme_policy_learning/scripts/train.py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path, help='Physical Intelligence pi05_base checkpoint root')
    parser.add_argument('dataset', type=Path, help='Prepared train subset with meta/stats.json')
    parser.add_argument('assets', type=Path, help='Directory containing robomme/norm_stats.json')
    parser.add_argument('output', type=Path, help='Parent directory for training runs')
    args = parser.parse_args()
    baseline = get_config('pi05_baseline')
    trainable = PathRegex(r'PaliGemma/llm/.*_1/.*|action_(in|out)_proj/.*|time_mlp_(in|out)/.*')
    cfg = replace(
        baseline,
        exp_name='movecube-train4',
        freeze_filter=nnx.Not(trainable),
        weight_loader=CheckpointWeightLoader(str((args.base / 'params').resolve())),
        data=replace(baseline.data, assets=replace(baseline.data.assets, assets_dir=str(args.assets.resolve()))),
        dataset_path=str(args.dataset.resolve()),
        checkpoint_base_dir=str(args.output.resolve()),
        batch_size=4, fsdp_devices=2, num_workers=0, num_train_steps=32, log_interval=1,
        lr_schedule=replace(baseline.lr_schedule, warmup_steps=4, peak_lr=2e-5, decay_steps=32, decay_lr=2e-6),
        seed=0, ema_decay=None, wandb_enabled=False,
    )
    logging.basicConfig(level=logging.INFO)
    train.main(cfg)


if __name__ == '__main__':
    main()
