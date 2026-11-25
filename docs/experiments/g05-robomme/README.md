# G05：RoboMME 四任务两阶段训练

目标是测量 G05 在 Panda 关节控制下的任务完成率。使用 `MoveCube`、`InsertPeg`、`PatternLock`、`RouteStick` 四项本地可用的官方训练数据，不把它们的分数推广到整个 RoboMME。环境端继续使用 `benchmarks/robomme/`。

## 数据和预注册的评估方法

HDF5 来自 `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337`。每项 100 条官方 `train` 轨迹。按完整 episode 划分：训练 90 条，离线验证使用编号 `0,11,22,33,44,55,66,77,88,99` 的 10 条。两阶段共用该划分和仅由训练集计算的 1%/99% 分位数。独立仿真预先固定官方 `test` 的 `0,2,3`（容易、中等、困难）三个原生回合，每个任务分别记录成功数和原生终止状态。选模型只用离线验证集。

原始 HDF5 含示范视频前缀和执行段。本次训练当前帧基线，只从执行段生成目标。训练输入采用前视和腕部 RGB、七个关节角、第一根夹指的位置及任务文本。核心网络每四帧取一个观测，预测后续连续 32 个原生动作（七个绝对关节目标及一个夹爪命令）。末尾只在当前轨迹内补齐，核心 loss 带 padding mask。编解码器从完整 32 步窗口采样。每次转换都比对任务和 episode seed；已逐项从原始 HDF5 回读第 11 轨迹，检查两路 RGB、关节状态和动作。

在本仓库根目录设置路径，`ART` 指向数据盘，不要把权重放进 Git：

```bash
export G05="$PWD/policy/G05/G05" PY="$PWD/policy/G05/G05/.venv/bin/python"
export EXP="$PWD/docs/experiments/g05-robomme" ART=/data/haichao-g05-robomme
export PYTHONPATH="$EXP:$G05:$G05/src:$PWD/.." OMP_NUM_THREADS=2
python "$EXP/prepare.py" ../datasets/robomme_data_h5 \
  ../robomme_benchmark/src/robomme/env_metadata "$ART/data"
```

基础训练包为 `OpenGalaxea/g05-robodojo@bb6cf7e45d0b2e8bd71c16c52961a93cc6125283` 中的 `g05_robodojo_train_base_assets.tar`，SHA256 `188fc8f092aa3ed2f67fcdb4a3e94ddbb44d4e0c41309e9f672a5a8594291218`。已对整包校验，并将其 `pretrained/` 内容解至 `$ART/base/`。其中 `action_tokenizer_hf/` 是第一阶段起点，`checkpoints/model_state_dict.pt` 是核心模型起点，`hf_processor/` 是匹配的视觉语言处理器。两阶段都从官方权重开始，不使用双臂 RoboDojo 成品模型做 Panda 初始化。

## 训练、选模与仿真

1. `codec.py` 从官方 ActionCodec 启动，适配编码器与解码器，固定量化字典以保留原有 token 编号。夹爪沿用上游二值规则。训练 1024 次，batch 24，AdamW 学习率 `2e-5`；每 128 步用同一组 384 个留出窗口评估 **两层残差码** 的七关节重建 MAE（弧度），在基础权重和历次训练权重中选误差最低的一次。
2. `core.py` 使用选出的 tokenizer 和同一批训练示范，按上游联合目标训练离散动作（CE）和连续动作（FM）。每张 4090 只有 24 GiB；这里更新 VLM 和动作专家各自最后 4 层，以及本体状态与动作投影，共 343,294,964 个参数；视觉主干冻结。可训练权重保留 FP32，前向用 BF16。使用均匀随机窗口，默认处理相当于训练集三遍的样本量。每 256 步计算固定 128 个留出窗口的联合 loss；按该指标保存 `best.pt`，并将当前参数、优化器及步数存入 `resume.pt`。中断后用相同的 GPU 数、batch 和总步数加 `--resume`。两种 loss 量纲不同，不把其和解释成成功率。
3. 基线和微调权重均使用这次训练集的归一化统计；独立服务用相同种子启动，都取连续动作头，执行每个 32 步预测的前 16 步。每个回合交给原生环境终止和判分，预算 1300 步。协议、模拟器、动作打包和评分不为 G05 特化。

```bash
CUDA_VISIBLE_DEVICES=3 "$PY" "$EXP/codec.py" "$ART"
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=3 "$EXP/core.py" "$ART"
for kind in base tuned; do
  "$PY" "$EXP/export.py" "$ART" "$kind"
  POLICY_GPU=2 ENV_GPU=3 bash "$EXP/evaluate.sh" \
    "$ART/eval-$kind/model_state_dict.pt" "$PWD/runs/robomme/g05-$kind"
done
```

`run/dataset_stats.json`、`run/action_tokenizer.pt`、`run/.hydra/config.yaml` 和选中 checkpoint 是推理所需的配套文件。导出器将增量核心参数合回官方基础权重。官方 loader 会将基础模型 27 维接口的相应投影截到任务使用的 20 维内部布局；Panda 的真实动作占其中 8 维（右臂 7 维＋夹爪 1 维），其余维度均有 mask。观察历史长度为 1，因此本轮不测试记忆增强版 G05。

已用真实样本验证两项核心 loss 的反向传播，并在单独的临时目录验证双卡优化、验证与保存。用实验分支的现有 `policy/G05/setup_eval_env_client.sh` 启动 `MoveCube/val/0`，预算 2，G05 通过 WebSocket 在真实模拟器执行 3 步并得到原生 `timeout`。这是通信与动作映射测试，不计入效果对照。32 项仓库回归通过，包括新加的单、双臂状态/动作往返及相机映射测试。
