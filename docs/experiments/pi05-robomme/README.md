# 从 π0.5 基础权重微调 MoveCube

实验从 Physical Intelligence 发布的 `gs://openpi-assets/checkpoints/pi05_base` 开始。数据来自官方 `Yinpei/robomme_data_h5` 的 MoveCube 训练集，使用 Panda 的 7 维绝对关节目标和 1 维夹爪命令。模型使用 RoboMME 的 `pi05_baseline`（无历史分支）及其随源码发布的 RoboMME 归一化统计。环境评估继续使用 `benchmarks/robomme/`。

以下命令在 XPolicyLab 根目录执行。`RUNTIME` 是 RoboMME 作者的 `RoboMME/robomme_policy_learning` checkout，本次固定为 `ac846a9a4aec841e483b1dc5d620dfe426bde853`。`PY` 是装好该运行时依赖的 π0.5 解释器，`ART` 用于独立存放数据和权重。

```bash
export RUNTIME="$PWD/policy/Pi_05/robomme_runtime"
export PY="$PWD/policy/Pi_05/openpi/.venv/bin/python"
export ART=/data/haichao-pi05-robomme
export PYTHONPATH="$RUNTIME/src:$RUNTIME/scripts:$PWD:$PWD/.."
RUNNER=docs/experiments/pi05-robomme
```

若从新工作区复现，先安装仓库原有 `policy/Pi_05` 环境，并按[环境文档](../../g05-pi05-robomme.md)准备 Panda 和仿真器。然后补齐 RoboMME 运行时。本次使用 Python 3.11、JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13；两张 24 GiB 的 RTX 4090 训练。若已有源码，核对下方 revision 即可。

```bash
git clone https://github.com/RoboMME/robomme_policy_learning.git "$RUNTIME"
git -C "$RUNTIME" checkout --detach ac846a9a4aec841e483b1dc5d620dfe426bde853
uv pip install --python "$PY" 'hydra-core==1.3.2' 'tqdm-loggable==0.2'
mkdir -p "$ART/assets/robomme"
cp "$RUNTIME/assets/norm_stats.json" "$ART/assets/robomme/norm_stats.json"
```

## 训练样本

原始 HDF5 版本为 `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。先确认任务名、训练 split 和 episode seed，然后提取执行阶段的两路原生 RGB、7 个关节位置、一个夹指位置、任务指令和后续 20 步动作。动作打包和末尾补齐使用上游的 `get_action_chunk`。无历史模型不读取示范帧的视觉特征，无需生成该缓存。输出目录必须是新的，避免混用两次准备的数据。

如还没有 HDF5，下载该版本的 `record_dataset_MoveCube.h5.tar.xz`（SHA256：`7ec562f822785ae61ca5073cedb35ed1900b06efc370108d423cc70aff86745f`），解压到 `../datasets/robomme_data_h5/`。该 HDF5 解压后约 26.3 GB。

```bash
"$PY" "$RUNNER/prepare_subset.py" \
  ../datasets/robomme_data_h5/record_dataset_MoveCube.h5 \
  ../robomme_benchmark/src/robomme/env_metadata/train/record_dataset_MoveCube_metadata.json \
  "$ART/movecube-train4"
```

| 训练 episode | seed | 首个执行帧 | 执行样本数 |
| --- | ---: | ---: | ---: |
| 0 | 14000 | 234 | 191 |
| 1 | 14100 | 258 | 183 |
| 2 | 14200 | 243 | 235 |
| 3 | 14300 | 161 | 105 |

这 714 个样本已通过上游 dataloader 的首批验证。batch size 为 4，两路图像分别为 `(4,224,224,3)`；归一化并填充到模型内部维度后，状态为 `(4,32)`，动作为 `(4,20,32)`，动作均为有限值。独立仿真评估采用 `MoveCube/test/0`，其种子为 `640000`。

## 初始化和训练

下载 Physical Intelligence 的 `pi05_base`。已有完整文件时直接传入其目录。上游提供 `scripts/download_pi05_base.py`；设置 OpenPI 缓存目录后，该脚本从公开 GCS bucket 下载。将 tokenizer 放在该缓存的 `big_vision/paligemma_tokenizer.model`，或让 OpenPI 首次使用时下载。下面将下载和训练分别调用。若本机的 tokenizer 已在默认缓存中，训练时可以继续使用原缓存，基础权重目录作为显式参数传入。

```bash
OPENPI_DATA_HOME="$ART" "$PY" "$RUNTIME/scripts/download_pi05_base.py"
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 \
  "$PY" "$RUNNER/finetune.py" "$ART/openpi-assets/checkpoints/pi05_base" \
  "$ART/movecube-train4" "$ART/assets" "$ART/checkpoints"
```

短训复用固定版本的上游 JAX `train.main`、flow-matching loss、数据变换和 checkpoint 写入。更新动作专家、动作输入输出投影和时间投影，视觉和语言主干冻结。由实际模型结构核对，基础权重覆盖全部 51 个参数张量，选中 19 个张量，共 430,098,464 个可训练参数，模型总参数为 3,353,433,872。两张卡做 FSDP，global batch 为 4，训练 32 次更新，seed 为 0。AdamW 与梯度裁剪沿用上游，学习率经过 4 步 warmup 达到 `2e-5`，随后余弦衰减至 `2e-6`。关闭 EMA 便于直接检验更新后的参数，关闭在线实验日志。最后一次更新的上游目录编号是 `31`。该版本保存可推理的参数和归一化统计，未保存优化器状态。

实际训练前按 GCS 对象元数据逐个核验 CRC32C：29 个对象，合计 12,441,749,581 字节。初始化时未出现缺失权重。32 个训练 batch（共计 128 个样本位置）全部完成，梯度范数在 0.1718–0.7085 之间；前 4 / 后 4 个 batch 的平均训练 loss 分别为 0.0903 / 0.0489。它们来自不同 batch，只作为优化过程的记录。

保存后逐个对比了全部 51 个参数张量：19 个可训练张量均有更新，其中 `action_out_proj.kernel` 的 32,768 个值全部变化，最大绝对差为 `2.948e-4`。其余 32 个冻结张量与基础权重按上游规则转换为 BF16 后逐元素相等。输出路径为 `$ART/checkpoints/pi05_baseline/movecube-train4/31/`，包含 `params/` 和 `assets/robomme/norm_stats.json`。再次运行请使用新的输出根目录。

## 独立测试回合与复现

在 `MoveCube/test/0` 上分别启动两个全新的策略进程，使用相同的 RoboMME 统计、模型随机种子 `0`、环境种子 `640000` 和 `action_steps=16`。只有权重变了。两次 reset 都返回 247 帧。下表保留模拟器的实际步数：传入预算为 1300，终止时原生计数为 1301。

| 权重 | 成功回合 | 原生终止状态 | 步数 |
| --- | ---: | --- | ---: |
| 官方 `pi05_base` | 0/1 | `timeout` | 1301 |
| 从同一权重微调 32 步 | 0/1 | `timeout` | 1301 |

这次验证了优化器确实更新目标参数，保存的 checkpoint 也能重新加载并驱动仿真。这个测试回合没有观察到成功；单个回合估计不了成功率。尤其要区分历史能力：`MoveCube` 要求重现视频里的移动方式，而本实验用的是只根据当前观测预测的 `pi05_baseline`。调整任务表现时，可以先固定一组验证回合，再比较训练量及是否提供演示历史。

复用的 Panda 模型服务来自已推送的实验提交 `59fe9968`。只在临时目录提取那份服务；环境始终从本分支的 `benchmarks/robomme/` 启动。以下命令沿用前文的 `ART`、`PY`、`RUNTIME` 和父工作区的 Panda 配置；本机示例用 GPU 2 跑推理、GPU 3 跑仿真，按可用设备调整。先将基础参数与训练使用的归一化统计放在同一推理入口下（软链接不复制权重）：

```bash
BASE="$ART/base-with-robomme-stats"
mkdir -p "$BASE"
ln -sfnT "$ART/openpi-assets/checkpoints/pi05_base/params" "$BASE/params"
ln -sfnT "$ART/assets" "$BASE/assets"
TUNED="$ART/checkpoints/pi05_baseline/movecube-train4/31"
REF=$(mktemp -d)
MODEL="$REF/XPolicyLab"
mkdir "$MODEL"
git fetch origin feat/robomme-g05-pi05
git archive 59fe9968 XPolicyLab.py model_template.py setup_policy_server.py utils client_server \
  policy/Pi_05/model.py policy/Pi_05/robomme.py policy/Pi_05/deploy.yml | tar -x -C "$MODEL"
ln -s "$PWD/../env_cfg" "$REF/env_cfg"
ln -s "$RUNTIME" "$MODEL/policy/Pi_05/robomme_runtime"
```

在同一终端运行以下函数。每次使用新结果目录；它等待策略服务就绪后才启动客户端，结束时关闭该进程。完整原生报告和 MP4 位于各自目录。

```bash
eval_one() (
  set -e
  ckpt=$1 out=$2
  test ! -e "$out"
  port=$(bash utils/get_free_port.sh)
  CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 PYTHONPATH="$MODEL:$REF" \
    "$PY" "$MODEL/setup_policy_server.py" --config_path "$MODEL/policy/Pi_05/deploy.yml" \
    --port "$port" --overrides "model_path=$ckpt" bench_name=RoboMME task_name=MoveCube \
    env_cfg_type=panda action_type=joint seed=0 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
  bash utils/wait_for_policy_server.sh localhost "$port" "$pid" Pi05 600
  EVAL_ENV_TYPE=sim ROBOMME_SPLIT=test ROBOMME_START_EPISODE=0 ROBOMME_EPISODES=1 \
    ROBOMME_MAX_STEPS=1300 ROBOMME_ACTION_STEPS=16 ROBOMME_OUTPUT_DIR="$out" \
    bash policy/Pi_05/setup_eval_env_client.sh RoboMME MoveCube external panda joint 0 3 \
    "$PWD/../robomme_benchmark/.venv" ckpt_name=external,action_type=joint "$port" localhost
)
eval_one "$BASE" "$PWD/runs/robomme/movecube-repeat-base"
eval_one "$TUNED" "$PWD/runs/robomme/movecube-repeat-tuned"
```

[实验记录](results.json)摘自实际日志和原生 `report.json`。本机原始训练日志、逐张量对比、仿真报告和录像在 `runs/robomme/finetune-pi05-movecube/`。Git 仅同步脚本、文档和小型结果记录。
