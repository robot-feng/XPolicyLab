# KnowAct 在 RoboMME 上逐级验证拟合能力

每轮同时记录专家观测上的动作预测、见过种子的自主执行、独立验证种子的自主执行。专家回放用于验证动作接口。成功由原生环境判定。所有回合使用正确的原生开场示范，除非明确标注消融。

| 实验 | 拟合的数据 | 实际要验证的能力 |
|---|---|---|
| 单轨迹 | PatternLock 官方 train/1 | 能记住动作，也能在自己的动作改变下一帧后重现整条路径 |
| 单任务 | PatternLock 官方 train/0–99 | 所有轨迹都参与梯度更新；同时观察训练种子和新种子的完整成功与第一段成功 |
| 多任务 | 官方 16 个任务各 100 条 | 逐任务比较单任务与联合训练，报告各任务及宏平均成功率，最后使用独立 test |

选择每一级的模型时看原生闭环，同时保留动作误差来定位失败。RoboMimic 对训练指标和部署目标差异做了实证分析。RoboMME 原论文发现不同任务受益于不同记忆表示，因此扩展时要保留逐任务成绩。[RoboMimic](https://proceedings.mlr.press/v164/mandlekar22a.html) · [RoboMME 论文](https://arxiv.org/abs/2603.04639) · [官方数据清单](https://huggingface.co/datasets/Yinpei/robomme_data_h5)

## 单轨迹

从已修复输入的 90 条轨迹、512 步 PatternLock 策略出发，另开优化器，只拟合 train/1。该轨迹包含 63 个动作、16 个 RGB 起点，官方种子为 15100。保持冻结的第一阶段 Panda ActionCodec、4 帧示范、32 步动作预测、训练时每 16 步一轮历史。四卡每卡一个窗口，全局 batch 为 4。第 200 步的训练 batch 动作 token 交叉熵为 0.0000535，旧划分中 10 条留出示范的固定 32 窗口 CE 为 1.2031。实验继承的 30K 四任务模型已经接触过其他训练示范。

| train/1，每次实际执行的动作数 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| 512 步基础策略 | 失败 | 失败 | 失败 | 失败 |
| 单轨迹再训练 200 步 | 失败 | 成功，44 步 | 失败 | 成功，68 步 |

上述每组固定初始观测和示范，并在每轮向模型回执实际执行动作数，原生环境提供后续观测。train/1 的成功路径在第 1、18、44 步依序触碰 4、5、1。每 16 步重规划的重复对照都在原步数失败。另起进程每 8 步重规划，同一单轨迹权重在 train/1 第 50 步失败。两次的示范、初始画面、首 8 个动作和第 8 步关节都相同，从第二次决策开始给出不同动作。现有记录不能区分该次决策时的渲染差异与模型计算差异。另一次官方 val/4 闭环按 0、1、4、3 的顺序走完，共 218 步。[train/1 成功录像](patternlock-train1-success.mp4) · [val/4 成功录像](patternlock-val4-success.mp4)

另外固定官方 val/4–7，单轨迹权重每 8 步重规划为 1/4，每 16 步为 0/4。512 步基础策略在相同四个种子上两种步长都为 0/4。这批种子未用于挑选步长。[逐回合评分与触碰顺序](single-trajectory-rollouts.csv)

### 定位误差

- 原始专家关节动作在 train/1、2、3 分别于 54、120、141 步成功。单轨迹策略在 train/1 使用专家观测与专家动作历史预测全部动作后，将该序列一次性送入仿真，第 57 步成功。真值动作经编解码再执行也在第 57 步成功。这两个回放都使用了专家信息。
- train/1 上单轨迹模型首 16 步的七关节预测 MAE 为 0.00452 rad。执行后末端相对专家偏了 2.20 mm，再过 16 步相差 10.31 cm。在固定历史的第二次决策上，真实闭环输入的关节 MAE 为 0.03115 rad；单独换回专家相机图像为 0.00582，单独换回专家关节状态为 0.03206。原 512 步模型在同一处更依赖关节输入。以上误差按该时间段计算，随权重和场景变化。
- 原 512 步模型在 5 条轨迹的相同专家观测上测试块内自回归。后续块中 770 个可变 token 由模型自行生成前块时命中 558 个，强制给定专家前块时命中 644 个。对这 5 条轨迹首轮之后的 30 个锚点等权平均，专家执行历史、自身预测历史、去掉执行历史三种条件的关节 MAE 依次为 0.02561、0.02876、0.04055 rad。这里保留了同一开场示范。
- 90 条训练轨迹有 72 种方向序列，10 条离线留出中有 6 条的完整方向组合从未出现。4 帧示范至少命中每一段连续同向移动的轨迹为 73/90，改取 8 帧为 88/90，16 帧为 90/90。这是离线时间覆盖统计，尚未测得增加示范帧带来的闭环收益。

### 复现

先执行[环境变量与数据准备](patternlock-pilot.md#数据准备与时序验收)。下列命令从 checkout 根目录运行，输出目录应为新路径。训练、离线探针和原生环境分别使用各自的解释器。

```bash
export PYTHONPATH="$EXP:$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/.."
export SINGLE_ROOT="$ART/single-trajectory/checkpoints"
export SINGLE="$SINGLE_ROOT/RoboMME-knowact-one-train1-panda-joint-7"
KNOWACT_OUTPUT_ROOT="$SINGLE_ROOT" KNOWACT_MAX_STEPS=200 bash policy/KnowAct/train.sh \
  RoboMME knowact-one-train1 panda joint 7 0,1,2,3 --task PatternLock \
  --train-episode 1 --warm-start "$NEW/last.pt" --save-every 200

# 原生服务与环境客户端；输出含 report.json 和按步录像。
KNOWACT_CKPT_PATH="$SINGLE/best.pt" KNOWACT_ALLOW_STRIDE_SHIFT=true KNOWACT_REPLAN_STEPS=8 \
  KNOWACT_CONTEXT_MAX_TOKENS=65536 EVAL_ENV_TYPE=sim ROBOMME_SPLIT=train \
  ROBOMME_START_EPISODE=1 ROBOMME_EPISODES=1 ROBOMME_MAX_STEPS=1300 ROBOMME_ACTION_STEPS=8 \
  ROBOMME_OUTPUT_DIR="$PILOT/repro-single-8" bash policy/KnowAct/eval.sh \
  RoboMME PatternLock knowact-one-train1 panda joint 7 0 0 "$KNOWACT_PYTHON" "$PWD/../robomme_benchmark/.venv"

export D="$PILOT/repro-diagnosis"
mkdir -p "$D"
"$ROBOMME_PYTHON" "$EXP/trace_patternlock.py" --data "$KNOWACT_ROBOMME_DATA" \
  --out "$D/expert" --episodes 1 2 3 --capture-inputs
"$KNOWACT_PYTHON" "$EXP/probe_history.py" --data "$KNOWACT_ROBOMME_DATA" \
  --weights "$SINGLE/best.pt" --episodes 1 --out "$D/offline"
"$ROBOMME_PYTHON" "$EXP/trace_patternlock.py" --data "$KNOWACT_ROBOMME_DATA" \
  --predictions "$D/offline" --out "$D/grafts" --episodes 1
"$ROBOMME_PYTHON" "$EXP/demo_coverage.py" --data "$KNOWACT_ROBOMME_DATA" \
  --h5 ../datasets/robomme_data_h5/record_dataset_PatternLock.h5 --out "$D/coverage.json"
```

同输入的步长对照需先启动策略服务。将服务端最大返回步数设为 32，再由诊断客户端裁剪实际执行的动作块，并向服务端回执。第二次决策探针用原生回放的画面和关节，重现前两块预测后单独替换每一种输入。

```bash
port=$(bash utils/get_free_port.sh)
KNOWACT_CKPT_PATH="$SINGLE/best.pt" KNOWACT_ALLOW_STRIDE_SHIFT=true KNOWACT_REPLAN_STEPS=32 \
  KNOWACT_CONTEXT_MAX_TOKENS=65536 bash policy/KnowAct/setup_eval_policy_server.sh \
  RoboMME PatternLock diagnosis panda joint 7 0 "$KNOWACT_PYTHON" "$port" localhost &
server=$!
bash utils/wait_for_policy_server.sh localhost "$port" "$server" KnowAct 300
CUDA_VISIBLE_DEVICES=0 "$ROBOMME_PYTHON" "$EXP/trace_patternlock.py" --data "$KNOWACT_ROBOMME_DATA" \
  --port "$port" --out "$D/paired" --episodes 1 --video \
  --conditions full_16 repeat_16 forget_16 full_4 full_8 full_32
kill "$server"
"$KNOWACT_PYTHON" "$EXP/probe_replan.py" --data "$KNOWACT_ROBOMME_DATA" \
  --weights "$SINGLE/best.pt" --reference "$D/expert" --live "$D/paired" --out "$D/second.json"
```

## 单任务全部示范

从同一个 512 步断点独立训练，使用官方 train/0–99，原 90/10 离线切分在此阶段合并。只使用训练集的优化器更新和固定 32 窗口拟合探针，指标名为 `train_probe_ce`。原生 val 的种子、示范和评分仅用于评估。保持示范帧数、编解码器、历史间隔和模型容量不变。每个 epoch 用 PyTorch DistributedSampler 跨四卡打乱排列，2,638 个不同窗口共取 2,640 次，末尾有 2 个 DDP 补齐窗口。每卡 batch 1，一个完整 epoch 为 660 次更新。恢复时按全局步数、原随机种子和 rank 跳到同一个位置。

```bash
KNOWACT_MAX_STEPS=660 bash policy/KnowAct/train.sh RoboMME knowact-pattern-full100 panda joint \
  7 0,1,2,3 --task PatternLock --all-train-demos --warm-start "$NEW/last.pt" --save-every 660
```

在查看该权重的结果前固定两组回合，train/0–7 测训练轨迹闭环、val/8–15 测新轨迹，两组都比较执行 8 步和 16 步。每次给出分母、第一段和整条路径。一个 epoch 只保证覆盖数据。若单任务训练种子也频繁失败，先对首次偏离处采集有正确标签的恢复样本，并用相同种子比较。若训练种子能稳定走完而验证落后，再分别比较 4/8 帧示范和数据多样性。观察到方向覆盖缺口，才值得用额外数据检验它的影响。

DAgger 给出了在策略自己访问的状态请求专家监督、聚合进已有训练数据的方法。这里专家必须能从实际偏离状态生成有效动作，将原轨迹时间对齐后的下一条动作照抄会引入错误标签。ACT 提供了预测动作块、用更频繁反馈控制长时误差的参照。当前步长对照只裁剪块和回执实际步数，尚未启用重叠预测的时间集成。[DAgger](https://proceedings.mlr.press/v15/ross11a.html) · [ACT 项目](https://tonyzhaozh.github.io/aloha/)
