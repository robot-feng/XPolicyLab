# TinyVLA

**Contributor:** RoboDojo Team | **Paper:** TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation | **arXiv:** https://arxiv.org/abs/2409.12514 | **Original code:** https://github.com/liyaxuanliyaxuan/TinyVLA

`TinyVLA` adapts the TinyVLA fast, data-efficient vision-language-action model to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `tinyvla/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

`install.sh` installs the adapter and XPolicyLab in editable mode into the active environment:

```bash
cd XPolicyLab/policy/TinyVLA
bash install.sh
conda activate <policy_env>  # e.g. tinyvla
```

## Data Processing

Converts RoboDojo HDF5 demos into `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`. `raw_task_dirs` is a source task directory or comma-separated list (unset converts all task directories under the source root); a non-numeric fifth argument is treated as `raw_task_dirs`. The source root defaults to `<repo>/data/<bench_name>` (`XPL_SOURCE_ROOT` override), and `TINYVLA_PROCESS_WORKERS` (8) / `TINYVLA_HDF5_COMPRESSION` (lzf) tune the conversion. If the output directory already exists, the script interactively asks whether to reuse it.

```bash
cd XPolicyLab/policy/TinyVLA
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert all RoboDojo demos for arx_x5 joint control
bash process_data.sh RoboDojo cotrain arx_x5 joint

# Example: create a 50-episode stack_bowls ablation
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

Both bundled `EpisodicDataset` implementations read HDF5 camera frames through the
adapter's `image_preprocessing.read_camera_images`. It calls the shared
`XPolicyLab.utils.process_data.decode_image_bit` before checking dimensions or resizing,
so XPolicyLab's marked JPEGs, legacy buffers and raw HWC uint8 RGB arrays use the same
boundary. The HDF5 `compress` flag does not determine color order. No channel swap follows
decoding. A singleton legacy container is accepted; multiple frames, invalid channels
and non-uint8 raw images are rejected with camera/frame context.

The existing raw-image resize rule is preserved: a decoded width unequal to `imsize`
resizes to 320×180 with OpenCV linear interpolation. Normalization and augmentation
remain in the original loader. Arbitrary unmarked standard JPEGs from other producers
are not distinguishable from legacy XPolicyLab buffers; convert external data with its
known source codec into RGB and use `encode_image_bit` for stored XPolicyLab image bits.

An offline diagnostic executes the actual `__getitem__`/index methods with real
Torch/HDF5 and synthetic images, comparing them with the retained pre-fix Git revision:

```bash
python tools/validate_tinyvla_image_boundary.py \
  --output runs/tinyvla-image-boundary/check
```

Run this from the repository root in an existing PyTorch environment. It requires the
default baseline `c378d3f04d68f41681077d97ea68b632598f9cd6` locally, or an explicit
`--baseline-ref`; a missing baseline is BLOCKED. The report retains vector-buffer shape
failures and a legacy red/blue reversal before the fix, along with current exact pixel
comparisons and source hashes. It bypasses the full loader constructor, augmentation and
language backbone. No checkpoint training, debug policy evaluation or simulator result
is established by this diagnostic.

```bash
cd XPolicyLab/policy/TinyVLA
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. The script snapshots `train.sh` and `deploy.yml` into the run directory, and if `<run_dir>/pretrained_vlm/` holds no weights it interactively offers to download a Llava-Pythia backbone (~400M / ~700M / ~1.3B for TinyVLA-S / B / H).

## Evaluation

```bash
cd XPolicyLab/policy/TinyVLA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `model_path`, `model_base`, `stats_path`, `enable_lora`, `conv_mode`, `camera_keys`.
