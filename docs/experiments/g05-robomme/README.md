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
