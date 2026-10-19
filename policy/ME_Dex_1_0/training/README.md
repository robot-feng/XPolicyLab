# Training

This directory contains the Clean50 data loader, configuration, and distributed training entry point for ME-Dex-1.0.

The recipe uses the unified tactile encoder with 16 RoboTwin surface slots, 48-dimensional latent vectors, and 18 temporal slices: 2 observed slices followed by 16 future slices. Set `topology` to `full_joint` or `h_bridge` in the YAML configuration.

Camera frames are returned as canonical RGB. The loader uses XPolicyLab's shared
`decode_image_bit` for raw arrays and both marked and historical JPEG encodings, with
no channel conversion in the training path. Checkpoints trained by this recipe should
use `input_color_order: rgb` at deployment. The separately released ME-Dex checkpoint
retains its documented `bgr` deployment setting.

Start a run with:

```bash
torchrun --nproc_per_node=16 -m training.train \
  --config training/configs/clean50_uni.yaml
```
