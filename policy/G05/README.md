# G05 RoboDojo Policy Adapter

**Contributor:** OpenGalaxea | **Paper:** Not applicable | **arXiv:** Not applicable | **Original code:** See vendored `G05/`.

This adapter serves G05 checkpoints through the XPolicyLab websocket policy-server interface. Integration scripts live at this directory level; the vendored public G05 implementation lives in `G05/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

For an isolated inference environment on RTX 4090 / Ampere / Ada, run from the
repository root (requires `uv`):

```bash
bash policy/G05/install.sh --inference
export G05_PYTHON="$PWD/policy/G05/G05/.venv/bin/python"
```

This creates a Python 3.10 environment from `requirements-inference.txt`, installs
the vendored source and XPolicyLab, and checks inference imports. It uses the
upstream SDPA fallback; FA4 and training/simulator packages are outside this
profile. Use the upstream full installation for training. Environment and native
checkpoint validation are documented in the [staged setup guide](../../docs/g05-pi05-robomme.md).

Install the XPolicyLab-side adapter dependencies:

```bash
cd XPolicyLab/policy/G05
export G05_PYTHON=/path/to/python
bash install.sh
```

The vendored G05 checkout is used by default. Set `G05_ROOT` only when using another compatible checkout:

```bash
export G05_PYTHON=/path/to/python
# optional:
export G05_ROOT=/path/to/another/G05_checkout
```

`G05_PYTHON` must point to an environment that can import the vendored G05 checkout and its dependencies. Checkpoint archives do not include a Python runtime.

The released RoboDojo FM checkpoint is verified with `arx_x5` / `joint` and the
matching official processor. Runtime dimensions and checkpoint selection use the
shared XPolicyLab helpers. Register parent-workspace metadata before an offline
check with `python scripts/setup_local_eval_metadata.py arx_x5` from the repo root.
The [staged guide](../../docs/g05-pi05-robomme.md) includes complete environment
variables and both plain/encoded RGB debug commands. `debug` accepts a venv prefix
or Python executable as eval argument 10; simulation retains its own environment recipe.

## Model Assets

Model files are hosted in:

```text
https://huggingface.co/OpenGalaxea/g05-robodojo
```

### RoboDojo simulation checkpoint

Use this archive for RoboDojo simulator evaluation:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_fm_only_checkpoint.tar \
  g05_robodojo_fm_only_checkpoint.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_sim

cd ./checkpoints/g05_robodojo_sim
sha256sum -c g05_robodojo_fm_only_checkpoint.tar.sha256
tar -xf g05_robodojo_fm_only_checkpoint.tar
```

The extracted checkpoint file is:

```text
hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint
```

Then set:

```bash
export G05_CKPT_PATH=/path/to/hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm
```

The public archive name intentionally does not encode the training step.

### RoboDojo-real checkpoint

Use this archive for RoboDojo-real evaluation only:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_real_checkpoint.tar \
  g05_robodojo_real_checkpoint.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_real

cd ./checkpoints/g05_robodojo_real
sha256sum -c g05_robodojo_real_checkpoint.tar.sha256
tar -xf g05_robodojo_real_checkpoint.tar
```

The extracted checkpoint file is:

```text
g05_robodojo_real_checkpoint/checkpoint/checkpoints/checkpoint.pt
```

Then set:

```bash
export G05_CKPT_PATH=/path/to/g05_robodojo_real_checkpoint/checkpoint/checkpoints/checkpoint.pt
export ROBODOJO_G05_ACTION_SOURCE=fm
```

This release does not include RoboDojo-real training code or data-processing scripts. The real checkpoint is provided for official RoboDojo-real evaluation with this adapter.

### Training base assets

Training base assets are separate from evaluation checkpoints:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_train_base_assets.tar \
  g05_robodojo_train_base_assets.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_train_assets

cd ./checkpoints/g05_robodojo_train_assets
sha256sum -c g05_robodojo_train_base_assets.tar.sha256
tar -xf g05_robodojo_train_base_assets.tar
```

The extracted directory contains:

```text
g05_robodojo_train_base_assets/pretrained/checkpoints/model_state_dict.pt
g05_robodojo_train_base_assets/pretrained/hf_processor/
g05_robodojo_train_base_assets/pretrained/action_tokenizer_hf/
```

Point the selected G05 training config to these assets using the config keys expected by that G05 checkout. RoboDojo datasets are not included in this archive.

## Data Processing

This adapter does not provide a standalone data converter. Use the official RoboDojo data download and conversion pipeline for simulator data. The policy reads the RoboDojo **LeRobot v3.0** dataset directly, from the root given by `ROBODOJO_LEROBOT_V30_ROOT`.

The keys are the standard ones of `XPolicyLab/scripts/transform_lerobot_v30_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)) — run that script to build a dataset for your own task subset. `G05/configs/data/robodojo.yaml` names them explicitly as `lerobot_key` entries: `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`, plus the flat `observation.state` and `action` vectors sliced by `start_index`. Nothing beyond the official output is required — no `meta/modality.json`, and normalization stats are computed during training.

## Training

G05 has separate ActionCodec training and core VLA training. This adapter's
`train.sh` launches **core VLA fine-tuning** using an existing codec checkpoint;
`fm_only`, `ar_only`, and `ar_fm` select objectives within that stage. They are
not the two training stages. The codec is selected by
`model.tokenizer.vq_config.ckpt_dir` / `G05_ACTION_TOKENIZER_PATH` and must match
the VLA run's action layout, vocabulary and normalization assets.

The vendored snapshot includes the codec model and reconstruction-loss code,
but has no standalone codec-training launcher (`train_vq.py`). Independent
codec training and core VLA training have **not been run** in this setup;
the verified inference profile is for evaluation. See the
[two-stage training notes](../../docs/g05-pi05-robomme.md#g05-两阶段训练边界) before training.

`train.sh` is a compatibility launcher around the vendored G05 checkout. It supports RoboDojo simulator training with `action_type=joint` and mode selection through `G05_TRAIN_MODE`.

Simulation training example:

```bash
export G05_PYTHON=/path/to/python
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
export G05_TRAIN_MODE=fm_only
export G05_MAX_STEPS=<num_training_steps>

cd XPolicyLab/policy/G05
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7 \
  <g05_training_overrides>
```

Supported `G05_TRAIN_MODE` values are `fm_only`, `ar_only`, and `ar_fm`. Training length is intentionally user-defined: set `G05_MAX_STEPS`, `G05_MAX_EPOCHS`, or pass `model.max_steps=...` / `model.max_epochs=...` as a Hydra override. Trailing arguments are passed through to the G05 training script.

Training length is not fixed by this adapter. For RoboDojo simulator training, users can start from a moderate number of optimizer steps and extend training if validation or simulator evaluation metrics continue to improve. The released checkpoint is intended as a reproducible evaluation baseline; further tuning of training length, data mixture, and model configuration may improve performance.

RoboDojo-real support in this release is evaluation-only. Use the released real checkpoint for official RoboDojo-real evaluation; real-world training code and data-processing scripts are not included here.

## Evaluation

Offline adapter check:

```bash
cd XPolicyLab/policy/G05
export EVAL_ENV_TYPE=debug
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=/path/to/extracted/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm

bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" base
```

Simulator-backed evaluation uses the same adapter entrypoint with `EVAL_ENV_TYPE` unset or set to `sim`:

```bash
cd XPolicyLab/policy/G05
unset EVAL_ENV_TYPE
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=/path/to/extracted/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm

bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" sim
```

For split-machine deployment via `setup_eval_policy_server.sh` and `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Notes

- Use `ROBODOJO_G05_ACTION_SOURCE=fm` for FM-style continuous action inference.
- Use `ROBODOJO_G05_ACTION_SOURCE=ar` only with checkpoints and G05 runtimes trained and validated for AR action decoding.
- For simulator evaluation, run from a RoboDojo checkout where evaluator-side `env_cfg/`, `scripts/`, `src/`, `task/`, and `Assets/` are available next to `XPolicyLab`.
