# PatternLock：开场示范的关节信息是否有用？

固定原生 RoboMME 模拟器、动作空间、初始权重和逐回合种子；完整成功以模拟器返回 `success` 为准。输入来自环境的开场示范和当前观测。目标按钮和已触按钮仅在评估结束后用于归因。未改 benchmark 客户端、动作顺序或判分。以下均为单任务诊断，不是 16 任务排行榜成绩。

## 冻结权重，改变示范读法

| 方法 | val/16–23 完整成功 | test/0–11 完整成功 | 手臂指令来源 |
|---|---:|---:|---|
| KnowAct，原 4 帧 RGB 示范 | 1/8 | 0/12 | 自回归动作解码 |
| 同一权重 + 拟合的关节动作头 | 8/8 | 12/12 | 完整示范的关节位置，结合当前关节估计进度；夹爪保持原预测 |
| 开环回放公开示范关节 | 8/8 | 未测 | 下一个示范关节位置；固定夹爪；完全不调用模型 |

[逐种子结果、目标和实际顺序](context-rollouts.csv) · 同一 test/0：[原权重录像](patternlock-base-test0.mp4) / [动作头录像](patternlock-demo-head-test0.mp4)。测试子集含 easy 6、medium 3、hard 3。基线 11 次失败、1 次超时。验证基线的 7 次失败都发生在路径第一段；完整命中要求按顺序走完整条路径。

环境在 reset 时交给策略双相机 RGB 和逐帧关节观测，包括开场示范。此前训练与部署只选取 4 帧示范 RGB；两端一致，但都丢弃了画面对应的关节观测。train/0 的 48 帧示范手臂关节与 HDF5 示范前缀逐帧一致。此信号和回放训练专家执行段的未来动作是两回事。这里的结论只适用于提供这类公开示范的任务和输入设置。

拟合时冻结 `knowact-pattern-full100`（既有 30K 四任务权重，再续训 PatternLock 512 + 660 步）。全部 100 条训练轨迹四卡分片提取预测；90 条拟合全局凸组合，10 条按 episode ID `% 11 == 0` 留作头部诊断，基座曾训练过这 10 条。每 8 步重规划；窗口起点来自专家观测，历史写入当次预测并确认的指令。候选分别是模型预测、示范领先 0/2/4/8 帧的关节位置。拟合所得权重约为 **0/0.275/0.725/0/0**。10 条内部留出的手臂绝对误差从 0.02782 降至 0.00142 rad。权重和原始指标见[模型头参数](demo-head-patternlock.json)。这说明关节轨迹足以解这个任务；它没有证明视觉主干学会了理解示范，也没有证明扰动后仍能纠偏。部署默认关闭此动作头，评测其它任务时必须显式选择，不能把这个结果记在原生模型名下。

## 复现

在仓库根目录执行。依赖、初始权重和对齐缓存见[训练和数据说明](patternlock-pilot.md)，全 100 条的基座见[逐级拟合](fit-ladder.md)。设置 `EXP`、`PILOT`、`KNOWACT_PYTHON`、`ROBOMME_PYTHON`、`KNOWACT_ROBOMME_DATA` 后：

```bash
BASE="$PILOT/checkpoints/RoboMME-knowact-pattern-full100-panda-joint-7/last.pt"
H5="$PWD/../datasets/robomme_data_h5/record_dataset_PatternLock.h5"
WORK="$PILOT/context-repro"; mkdir -p "$WORK"
export PYTHONPATH="$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/.."
export KNOWACT_REPLAN_STEPS=8 KNOWACT_ALLOW_STRIDE_SHIFT=true KNOWACT_CONTEXT_MAX_TOKENS=65536
unset KNOWACT_DEMO_HEAD
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g "$KNOWACT_PYTHON" "$EXP/fit_demo_head.py" collect --weights "$BASE" \
    --data "$KNOWACT_ROBOMME_DATA" --h5 "$H5" --rank "$g" --world 4 --output "$WORK/rank$g.npz" &
done
wait
"$KNOWACT_PYTHON" "$EXP/fit_demo_head.py" fit --parts "$WORK"/rank*.npz --output "$WORK/head.json"
CUDA_VISIBLE_DEVICES=0 "$ROBOMME_PYTHON" "$EXP/demo_replay_control.py" \
  --split val --start 16 --count 8 --output "$WORK/replay.json"

# 默认原模型。复现动作头，在命令前 export KNOWACT_DEMO_HEAD="$WORK/head.json"。
# 每次使用新输出目录。改为 split=test/start=0/episodes=12 可复现测试子集。
EVAL_ENV_TYPE=sim KNOWACT_CKPT_PATH="$BASE" ROBOMME_SPLIT=val ROBOMME_START_EPISODE=16 \
  ROBOMME_EPISODES=8 ROBOMME_MAX_STEPS=1300 ROBOMME_ACTION_STEPS=8 ROBOMME_OUTPUT_DIR="$WORK/base-val" \
  bash policy/KnowAct/eval.sh RoboMME PatternLock comparison panda joint 7 0 0 "$KNOWACT_PYTHON" "$(dirname "$(dirname "$ROBOMME_PYTHON")")"
```

## 完整测试、跨任务和采样密度

冻结上表同一模型头，继续测试剩余 38 个 PatternLock 官方 test 回合。合并 test/0–49 共 **50/50**：easy 26/26、medium 12/12、hard 12/12。原模型的配对结论仍限于 test/0–11 的 0/12；另外 38 个种子未跑原模型。将两种策略原样切换到另一个模仿任务 RouteStick，不在该任务重新拟合：val/0–5，原权重 **0/6**（5 次 fail、1 次 timeout），同一动作头 **6/6**。这是单任务动作头的迁移；其冻结基座的 30K 预训练阶段用过 RouteStick。[逐种子结果](context-transfer.csv) · [RouteStick 验证录像](routestick-demo-head-val2.mp4)。可复用上面的评估命令，替换任务名、split 和 episode 范围，并为每次实验指定新目录。

另取上述相同的 8 个 PatternLock 验证种子，不使用模型，只改变公开示范中保留的关节观测数量。保存两端点并均匀抽帧，沿原时间轴在关节空间线性插值，使用相同执行时钟、固定夹爪和 1,300 步预算。只保留 4 帧成功 **4/8**，保留 16 帧 **8/8**，完整示范 **8/8**。[逐回合采样对照](context-sampling.csv)。这说明均匀 4 帧确实会丢失部分回合所需的路径细节；它不能区分关节信息与视觉的独立贡献，也不能证明增加神经网络输入就能得到回放成绩。复现时在 `demo_replay_control.py` 的命令中分别追加 `--demo-frames 4`、`--demo-frames 16` 或省略该参数。

## 原生 KnowAct 能不能学会使用示范关节？

三组均关闭 `KNOWACT_DEMO_HEAD`，由原生 BAR 和 Panda ActionCodec 生成整段动作。固定 `knowact-pattern-full100` 起点、训练数据和打乱顺序，四张 RTX 4090 各 batch 1、全局 batch 4、种子 7、学习率上限 `2e-5`、256 次优化器更新。训练池是 PatternLock 的 100 条官方训练轨迹，共 2,638 个 RGB 窗口，256 步只使用其中 1,024 个窗口。固定配置只训练原有 342,333,408 个参数；不重新训练编解码器。所有组训练历史步长为 16，推理/执行每 8 步重规划；这项已有差异未随消融变化。CE 是最后断点在固定 **16 个训练窗口**上的动作 token 损失，不能代替原生成功率。

| 示范输入（神经网络生成动作） | 训练探针 CE | val 完整成功 | val 首段正确 | test 完整成功 | test 首段正确 |
|---|---:|---:|---:|---:|---:|
| 4 帧 RGB | 0.16482 | 1/8 | 2/8 | 0/12 | 1/12 |
| 4 帧 RGB + 4 个关节观测 | 0.17163 | 1/8 | 3/8 | 0/12 | 1/12 |
| 4 帧 RGB + 16 个关节观测 | 0.17445 | 0/8 | 1/8 | 0/12 | 2/12 |

验证和测试与前两节完全相同。首段正确要求前两个触碰按钮按顺序匹配；中途超时仍然算完整任务失败。四关节版本在验证集多走对了一个首段，未带来更多完整成功。[逐回合数据](context-native.csv)。

**一次应当剔除的试验。** 接入新示范状态时，归一化器会替换传入字典中的值。第一次取数据与部署逐项相同，第二次取同一条缓存却读到了归一化后的关节，train/0 首帧 7 个轴全变，最大绝对差 2.04221。已修正为使用独立的映射，并加入“重复取样不改变缓存”的校验。试跑所得 CE 0.19213 和验证 0/8 均不用于结论；上表四关节成绩是从原始权重重新训练的独立结果。这个错误是本轮原型引入的，不是此前 RGB 基线低成功率的原因。

训练输入只从 `t < first_execution` 的 HDF5 示范观测读取关节，不读动作标签作为输入。部署只读取当前 reset 时环境按帧送来的观测。更密的关节采样按时间分组附在原来的四个图像锚点上，不增加送入视觉编码器的图像数。每组使用已有的 proprio encoder，checkpoint 保存采样配置；旧权重继续走原来的 RGB 示范路径。

```bash
# 前文的 BASE、H5、EXP、WORK、两个 Python 和已有归一化资产均保持不变。
unset KNOWACT_DEMO_HEAD
for mode in rgb proprio4 proprio16; do
  extra=()
  if [[ $mode != rgb ]]; then extra+=(--demo-state-h5 "$H5"); fi
  if [[ $mode == proprio16 ]]; then extra+=(--demo-state-frames 16); fi
  "$KNOWACT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 "$EXP/train.py" \
    --initial "$KNOWACT_INITIAL" --data "$KNOWACT_ROBOMME_DATA" --output "$WORK/$mode" \
    --task PatternLock --all-train-demos --warm-start "$BASE" --steps 256 --batch 1 --seed 7 \
    --save-every 256 --val-windows 16 "${extra[@]}"
done
# 用前文模拟器命令分别指定 KNOWACT_CKPT_PATH="$WORK/<mode>/best.pt" 和新的输出目录。
# 严格核对 train/0 的重复读取、归一化值、掩码、示范像素以及真实模型输出。
# 此检查用断点原生 16 步，闭环成功率对照才选择 8 步。
env -u KNOWACT_REPLAN_STEPS -u KNOWACT_ALLOW_STRIDE_SHIFT "$KNOWACT_PYTHON" "$EXP/verify_aligned.py" check \
  --data "$KNOWACT_ROBOMME_DATA" --capture "$PILOT/capture.npz" \
  --weights "$WORK/proprio4/best.pt" --demo-state-h5 "$H5"
```

16 状态版本在 train/0 的真实回放确认视觉锚点仍为 `[0, 15, 31, 47]`，一共 16 个关节 token、5 次原生上下文追加（4 次示范 + 1 次当前观测）。训练与服务的 token、掩码、示范图像和关节值通过逐项校验，重复取样后原始关节不变。新权重在 WebSocket 原始 RGB 和编码 RGB 两条路径各完成一个 debug 回合；项目测试 32 项和 KnowAct/适配器测试 75 项通过。SHA-256、精确 CE 和验收指标见[可核验的记录](context-checks.json)。大权重留在实验机，GitHub 保存实现、可重现设置、拟合头、逐种子结果和代表录像。

**决策。** 对 PatternLock，用完整公开示范的动作头作为已验证的模仿控制基线。对原生 KnowAct，单纯增加上下文采样，在这次 256 步的配对实验中没有提高完整成功率，不据此启动相同目标的 30K 长跑。下一轮应固定已验证的输入通路，在训练种子上增加“当前该对齐示范的哪一段”的监督：同一当前观测配不同示范检验目标是否切换，同一示范配偏离后的观测检验是否会纠偏。相同测试种子已经用于诊断，新方案先比验证集完整成功、首段命中及偏离后的恢复，再用未参与选择的种子确认。其余任务，尤其没有公开示范或需要脱离示范的任务，仍需独立评估。
