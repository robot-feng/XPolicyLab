# 从 π0.5 基础权重微调 MoveCube

实验从 Physical Intelligence 发布的 `gs://openpi-assets/checkpoints/pi05_base` 开始。数据来自官方 `Yinpei/robomme_data_h5` 的 MoveCube 训练集，使用 Panda 的 7 维绝对关节目标和 1 维夹爪命令。模型使用 RoboMME 的 `pi05_baseline`（无历史分支）及其随源码发布的 RoboMME 归一化统计。环境评估继续使用 `benchmarks/robomme/`。

以下命令在 XPolicyLab 根目录执行。`RUNTIME` 是 RoboMME 作者的 `RoboMME/robomme_policy_learning` checkout，本次固定为 `ac846a9a4aec841e483b1dc5d620dfe426bde853`。`PY` 是装好该运行时依赖的 π0.5 解释器，`ART` 用于独立存放数据和权重。

```bash
export RUNTIME="$PWD/policy/Pi_05/robomme_runtime"
export PY="$PWD/policy/Pi_05/openpi/.venv/bin/python"
export ART=/data/haichao-pi05-robomme
export PYTHONPATH="$RUNTIME/src:$RUNTIME/scripts:$PWD:$PWD/.."
RUNNER=docs/experiments/pi05-robomme
mkdir -p "$ART/assets/robomme"
cp "$RUNTIME/assets/norm_stats.json" "$ART/assets/robomme/norm_stats.json"
```

## 训练样本

原始 HDF5 版本为 `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。先确认任务名、训练 split 和 episode seed，然后提取执行阶段的两路原生 RGB、7 个关节位置、一个夹指位置、任务指令和后续 20 步动作。动作打包和末尾补齐使用上游的 `get_action_chunk`。无历史模型不读取示范帧的视觉特征，无需生成该缓存。输出目录必须是新的，避免混用两次准备的数据。

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
