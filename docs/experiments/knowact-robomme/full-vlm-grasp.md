# MoveCube：从 KnowAct 30K 解冻整个视觉语言主干
本轮针对[基础抓放闭环失败](basic-grasp.md)，从四任务 KnowAct Qwen3.5-0.8B 的 `step_030000.pt` 热启动，再训练 30,000 个优化步。新优化器从 0 计数，旧权重保留。只使用 MoveCube 的官方 train 场景（含三种搬运方式），其中 90 条拟合、10 条离线留出；原生验证回合不进入训练。它是针对 MoveCube 的续训，不是全任务保持能力的验证。

- **参数**：24 层语言主干（含共享输入/输出词嵌入）和 12 层视觉编码器（含 patch embedding 和 merger）全部解冻；本体状态投影继续训练。Panda ActionCodec 沿用此前适配的版本，连续动作专家不参与 AR 损失。原训练默认范围仍然可用，显式传 `--full-vlm` 才启用全主干更新。
- **输入**：动作前对齐的双相机 RGB 和本体状态，开场示范均匀取 4 个时刻、只给两路 RGB。专家执行动作只作为执行阶段的监督或已执行历史，不作为开场示范输入。使用 Qwen 官方静态图像分块排列。相对于旧 30K，数据、视觉排列和可训练参数均有变化，不能把效果归因于解冻一个因素。
- **优化**：4 张 RTX 4090，单机 4 进程，每卡每步 1 个窗口、全局 batch 4，AdamW，100 步学习率预热、峰值 1e-5，梯度裁剪 1.0，BF16 自动混合精度，FP32 可训练参数。使用激活检查点压低显存。记录的 CE 是离散动作 token 的交叉熵；它不代替抓放成功率。
- **验收**：启动和恢复的第一步都检查全部选中参数有梯度，并检测视觉 patch 卷积、第一层语言 MLP 的权重确有变化。先保存第 4 步训练状态，再恢复以验证断点；第 128、1,000、10,000、20,000、30,000 步用当步的 `last.pt` 测试。固定 MoveCube train/1、4 和 val/1、8；分别记录物理抓握、抬起、放置和原生状态。没有专家动作输入或示范跟踪头。离线验证/保存还每 512 步执行一次，覆盖 `last.pt`，按留出 CE 保留 `best.pt`。早期闭环样本小，不当作基准成绩。

启动验收（2026-11-28）：四个 DDP rank 均参与，FLA 内核已启用；858,019,904 个不重复的可训练参数、479 个张量。第 4 步断点实际含 479 组 Adam 状态，固定 32 个留出窗口的动作 CE 为 1.56393。与旧权重比较，在视觉 patch 卷积、视觉第 12 层以及语言第 1/24 层分别抽查 8,192 个元素，均发生更新，冻结的动作专家样本未变。动作前对齐抽查了 train/1、4、25、50、75 的第 16 步：RGB、关节、动作标签与原始记录在训练存储精度下完全对应。第一组闭环评估排在新阶段第 128 步。阶段结果按四个原生回合的 JSON 同步到 [独立实验记录分支](https://github.com/robot-feng/XPolicyLab/tree/reports/knowact-movecube-fullvlm/docs/experiments/knowact-robomme/full-vlm-results)。

从 XPolicyLab 根目录，在已配置的本机重跑：

```bash
export ART=/data/haichao-knowact-robomme
export KNOWACT_INITIAL="$ART/initial-qwen-state" KNOWACT_OUTPUT_ROOT="$ART/checkpoints" KNOWACT_ROBOMME_DATA="$ART/preaction"
export OLD="$ART/checkpoints/RoboMME-knowact08-state-panda-joint-7/checkpoints/step_030000.pt"
export KNOWACT_PYTHON="$PWD/policy/KnowAct/.venv/bin/python" SIM_PY="$PWD/../robomme_benchmark/.venv/bin/python"
export PYTHONPATH="$PWD/docs/experiments/knowact-robomme:$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$SIM_PY" docs/experiments/knowact-robomme/prepare_aligned.py --task MoveCube --source /data/haichao-g05-robomme \
  --h5 "$PWD/../datasets/robomme_data_h5" --output "$KNOWACT_ROBOMME_DATA"
bash docs/experiments/knowact-robomme/full_vlm_grasp.sh
```

[调度脚本](full_vlm_grasp.sh)恢复同名目录的训练进度。运行中可查看 `runs/robomme/knowact/movecube-full-vlm/train.log`；每个评估阶段的逐回合 JSON、双相机视频在同目录的 `eval-<step>/`。权重和优化器留在 `$KNOWACT_OUTPUT_ROOT/RoboMME-knowact-movecube-fullvlm-panda-joint-7/`。基线和新模型使用同一组环境种子。
