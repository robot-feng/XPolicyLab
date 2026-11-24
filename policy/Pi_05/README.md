# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

For a checkpoint-loading and native-inference check before starting XPolicyLab
evaluation, see the [staged setup guide](../../docs/g05-pi05-robomme.md). It uses
the official `pi05_libero` checkpoint with the upstream OpenPI inference API;
this is independent of the adapter's default RoboDojo training configuration.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The dataset uses the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)); the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## RoboMME Evaluation

This evaluation backend uses the [official RoboMME policy code](https://github.com/RoboMME/robomme_policy_learning)
at `ac846a9a4aec841e483b1dc5d620dfe426bde853` and the
[RoboMME π0.5 baseline checkpoint](https://huggingface.co/Yinpei/pi05_baseline),
rather than the native ALOHA or LIBERO configuration. It accepts Panda joint control only:
front/wrist RGB, seven joint positions and the first physical finger position; it returns
20 absolute joint/gripper actions and executes 16 before replanning by default. Gripper action
units are `-1` closed / `+1` open. The official transforms handle normalization and delta-to-absolute
conversion. No adapter-side color conversion, dummy third camera, or gripper rescaling is applied.

From the repository root, after the native installation:

```bash
bash policy/Pi_05/install_robomme.sh
python scripts/download_pretrained_models.py pi05-robomme --parts 8
policy/Pi_05/openpi/.venv/bin/python -m zipfile -e \
  policy/Pi_05/checkpoints/robomme/79999.zip policy/Pi_05/checkpoints/robomme
# The archive includes the author's original directory tree.
mv policy/Pi_05/checkpoints/robomme/home/daiyp/MME-VLA-Suite/runs/ckpts/pi05_baseline/pi05/79999 \
  policy/Pi_05/checkpoints/robomme/79999
python scripts/setup_local_eval_metadata.py panda
```

Install RoboMME and its assets in a separate environment following its official guide. The
evaluation environment additionally needs XPolicyLab's dependencies (notably `pydantic>=2.5`
and `msgpack-numpy>=0.4.8`); no policy Torch/JAX dependencies need to be copied into it.
The local verified environment is `<workspace>/robomme_benchmark/.venv`.

```bash
cd policy/Pi_05
# Existing native venv `uv` hosts the policy; the final argument hosts the simulator.
ROBOMME_EPISODES=1 ROBOMME_MAX_STEPS=1300 bash eval.sh \
  RoboMME PickXtimes ./checkpoints/robomme/79999 panda joint 0 1 3 uv \
  "$(cd ../../../robomme_benchmark/.venv && pwd)"
```

Use `EVAL_ENV_TYPE=debug DEBUG_EVAL_EPISODE_NUM=1` for the same checkpoint's offline
contract check; repeat with `DEBUG_OBS_ENCODED=1`. Real RoboMME evaluation uses uncompressed RGB.
`ROBOMME_SPLIT` defaults to `test` (`train`, `val`, `ood_test` also supported),
`ROBOMME_START_EPISODE=0`, `ROBOMME_EPISODES=1` (or `all`),
`ROBOMME_MAX_STEPS=1300`, `ROBOMME_ACTION_STEPS=16`. Task names may be comma-separated or `all`.
`ROBOMME_OUTPUT_DIR` selects a **new** output directory; by default results go to
`runs/robomme/<timestamp-id>/` with per-episode JSON, aggregate JSON and front/wrist MP4.
Benchmark-owned seeds are preserved; the eval seed controls the policy RNG. Reset/demo frames
are forwarded in order. This baseline has no history memory and uses the latest frame.

This backend is evaluation-only: the existing data conversion and training scripts continue
to target the native OpenPI recipe. The pinned runtime is downloaded into the ignored
`robomme_runtime/` directory and never overwrites `openpi/`. Starting a fresh policy server
is required when switching backend. See the [detailed validation log](../../docs/g05-pi05-robomme.md)
for exact local versions and the distinction between interface smoke tests and task success.

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.
