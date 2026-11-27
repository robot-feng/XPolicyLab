# KnowAct 30K 能不能自主抓起并放下一个方块？

六个 easy 回合均在 1,301 次控制后超时，没有一次触发目标方块的物理抓握，也没有完成抬起或放置。最有诊断价值的是 MoveCube 的 train/1、train/4：这两个布局参加过 30K 训练，且各自的专家动作在同一个模拟器和起点都能抓放成功。这一版自主策略没有通过基础抓取检查。样本是有意筛出的抓放子集，不代表整套 RoboMME 的成功率。

[同步录像](grasp-train1-comparison.mp4)（上：KnowAct，下：该起点的专家；每行左前视、右腕部，20 fps，两路从同一次序的执行第 0 帧开始，到专家成功为止） · [逐回合原始数值](grasp-trials.csv) · [计分脚本](probe_grasp.py)。本机 `runs/robomme/knowact/basic-grasp-v1/` 中保存六个带有同名 JSON 的完整自主回合 MP4。

## 如何选题和计分

- **MoveCube** 有三种操作方式。按回合编号在 easy 场景中选择原生 `reset()` **执行完开场示范以后**的 `grasp_putdown`，避开另两种推方块的操作。构造环境时、正式 `reset()` 之前读到的方式可能不同。选已在历史回放中验证过的 train/1、4 作同起点对照，选验证集中按编号最先符合条件的 val/1、8；筛选不使用模型表现。任务按顺序要求抬起和放到目标，不需要按按钮。
- **PickXtimes** 选择验证集中按编号最先满足 easy、一个方块、一次抓放的 val/1、5。开场没有示范，指令分别指向蓝块、绿块。放到目标后还要按下按钮，故单列放置与整回合成功。这是跨任务检查：本次四任务 30K 权重训练过 MoveCube、PatternLock、RouteStick、InsertPeg，没有训练过 PickXtimes。
- 自主控制沿用 [RoboMME 环境客户端](../../../benchmarks/robomme/client.py) 和 KnowAct 的 AR 服务：双路 RGB、实测关节与夹爪、指令、已执行历史，每 16 步从真实新观测重规划。无示范跟踪动作头，无专家动作输入。旧权重的配置没有 `native_demo_frames`，其有序开场示范前缀默认为 0；MoveCube 的环境仍播放示范，这限制了该权重区分操作方式的能力。PickXtimes 的这些回合本来就没有视频。训练集回合用于检查拟合，验证回合从未作为此次训练数据。
- 每个执行步只在环境端记录：① Panda 的 `agent.is_grasping(目标方块)`；② 原生计分器是否走过 pickup（要求真正抓住且方块中心高度大于 5 cm）；③ 是否继续走过 placement；④ 模拟器终止 `status`。方块位置、目标、子任务与分数均不发给策略。工具中心点到方块中心的最小三维距离用于解释，不能替代抓握判定。每回合的 1,300 步上限沿用已有评估入口，原生 wrapper 的超时返回在第 1,301 次调用。

| 自主闭环，固定 step 30,000 | 物理抓住 | 抬起 | 放到目标 | 整回合成功 | 末端到方块最近距离（按回合顺序） |
| --- | ---: | ---: | ---: | ---: | --- |
| MoveCube train/1、4 | 0/2 | 0/2 | 0/2 | 0/2 | 3.82、6.07 cm |
| MoveCube val/1、8 | 0/2 | 0/2 | 0/2 | 0/2 | 9.25、8.20 cm |
| PickXtimes val/1、5 | 0/2 | 0/2 | 0/2 | 0/2 | 11.17、9.35 cm |

六个回合中实测双指总开度都从约 8 cm 到过 0 cm。夹爪在动，但合拢没有形成有效抓握。例如同一布局 train/1，专家第 80 步抓稳、第 94 步通过抬起判定、第 151 步放到目标；KnowAct 最接近时仍相隔 3.82 cm，专家为 0.53 cm。这里不能仅凭距离断言某个关节标定错误，或把失败归结于松手、计数、按钮。

## 把动作编解码、动作预测和闭环反馈拆开

以下四个对照都只用 MoveCube train/1、4，各自从同一个原生起点执行，由相同判定器计分。对照利用离线记录，**不能算模型自主成功率**。凡是回放动作的条件，模拟器都真实执行这些动作，但不会把执行后的画面送回去改变下一步预测。

| 抓放控制条件 | 下一段动作怎么来 | 两个回合的结果（train/1、4） |
| --- | --- | --- |
| 原专家动作 | 直接执行各自记录的执行阶段动作 | 均成功，第 151、171 步放对 |
| 专家动作经过 KnowAct 编解码 | 把专家动作编成该权重的动作 token，从实际部署解码路径还原、反归一化 | 均成功，第 151、171 步放对 |
| 专家观测、专家过去动作 | KnowAct 看记录里的 RGB 和本体状态，自行预测下一段；历史中填写专家已经执行的动作 | 均成功，第 151、171 步放对 |
| 专家观测、模型自己过去的动作 | 同样使用记录里的 RGB 和本体状态；历史改用模型前面预测的动作，无当前目标动作输入 | 均成功，第 151、172 步放对 |
| **自主闭环** | **模型自己控制；下次决策使用执行后的真实 RGB、本体状态及自己已执行的动作** | **均超时，均未抓住** |

这两条成功轨迹可以由当前编解码器表达，且在专家观测时间线上，模型能预测出可执行的抓放动作。只把过去的动作换成模型自己的预测，这两个对照仍能成功；最大的已观察差别出现在**要求模型对自身执行后产生的观测作出反应**时。另比对两条回合的原生执行起始观测与旧缓存首样本：两条指令相同，前视首帧均逐像素相同；腕部首帧平均像素绝对差分别为 0.0114 和 0（0–255），七个手臂关节最大差均小于 0.000001 rad。这只是首帧核验，不证明以后每次重规划的输入、时序和在线历史都一致。

## 同一个策略服务上，只换每次决策的观测

为了排除“离线用单条调用，线上用批量 WebSocket”这个混杂因素，用同一个在线服务、同一批量接口、同一环境与 16 步回执做配对。两边都真正驱动机器人，也都只保留机器人自己执行的动作历史。一边照常使用自己的新 RGB 和关节；另一边在每次决策时同时换成该训练回合对应时间的记录 RGB 和关节。后一条件不能作为自主评估。这个局部诊断最多执行记录的长度（183、204 步），不把到达诊断上限说成原生超时；自主成绩仍取上面的完整回合。

| 官方训练回合 | 给真实执行观测 | 只替换成记录观测 | 两种条件何时出现不同的关节命令 |
| --- | --- | --- | --- |
| train/1 | 前 183 步未抓住 | 第 81 步抓住，第 151 步放对 | 第一个动作块已经不同；前 16 步最大单关节差 0.06882 rad（3.94°） |
| train/4 | 前 204 步未抓住 | 第 83 步抓住，第 173 步放对 | 首块 16 步的七关节与夹爪命令完全一致；第 16 步重规划后，第二块最大单关节差 0.12122 rad（6.95°） |

再把**首个决策**单独拆成四种组合：两路图像取真实 reset 或记录首帧，本体状态也分别取两种来源。每次请求前清空策略历史，用同一 WS 批量接口、同一指令，重复发送两端的原样输入。train/1 中，前视像素完全相同，腕部 RGB 只有 2,139 / 65,536 个像素位置有变化，任一通道的最大差是 2 / 255。只换本体状态，七关节命令不变；只换 RGB 就复现全部 0.06882 rad 差异。train/4 的四种组合首块均相同。两回合原样重复的结果都复现。这说明这份权重在**特定真实首帧**上对很小的视觉变化敏感，不证明单张图或任何一种误差解释了所有失败。

[同服务配对脚本](probe_grasp_feedback.py) · [每个动作块的差值](grasp-feedback.csv) · [首帧输入替换原始记录](grasp-reset-inputs.csv)。首帧拆分可用保存的两份 `obs` 在同一服务上复核：每次依次调用 `reset()`、`update_obs_batch([obs])` 和 `get_action_batch([0])`，比较首 16 步七个关节。只替换 `vision` 或 `state`，其余字段不动，最后重复原样输入。

**下一步留在 train/1 和 train/4。** 固定成功的动作编解码器。先在 train/1 的首帧比较图像预处理和首段离散动作 token，找出像素微变从哪里放大；再在 train/4 第 16 步分别只换画面或只换关节，区分视觉与本体反馈。然后用动作前对齐的数据拟合一条轨迹，以相同起点多次独立重置、自主抓起并放到目标为验收。过这一关再换未训练布局。已有证据不支持直接加长训练，也不能把这份 30K 权重的行为外推为 KnowAct 架构学不会抓放。

## 复现

从已配置好的 XPolicyLab checkout 根目录运行，需父工作区 Panda `env_cfg`、RoboMME 原版模拟器、训练产物及其同目录 `.hydra` 和归一化统计。权重 SHA-256 为 `28b50b5b4d3118bd2a67ab5af3f9500196b81e3a622aeb5db41a78f5dc782cdb`。环境定义见上游 [MoveCube](https://github.com/RoboMME/robomme_benchmark/blob/main/src/robomme/robomme_env/MoveCube.py)、[PickXtimes](https://github.com/RoboMME/robomme_benchmark/blob/main/src/robomme/robomme_env/PickXtimes.py) 和[原生子目标判定](https://github.com/RoboMME/robomme_benchmark/blob/main/src/robomme/robomme_env/utils/subgoal_evaluate_func.py)。

```bash
EXP="$PWD/docs/experiments/knowact-robomme"
OUT="$PWD/runs/robomme/knowact/grasp-repro-$(date +%s)"
DATA=/data/haichao-g05-robomme
SIM="$PWD/../robomme_benchmark/.venv/bin/python"
export KNOWACT_PYTHON="$PWD/policy/KnowAct/.venv/bin/python"
export KNOWACT_CKPT_PATH=/data/haichao-knowact-robomme/checkpoints/RoboMME-knowact08-state-panda-joint-7/checkpoints/step_030000.pt
export KNOWACT_NATIVE_CONTEXT=true KNOWACT_KEEP_HISTORY=true KNOWACT_CONTEXT_MAX_TOKENS=65536 KNOWACT_REPLAN_STEPS=16
unset KNOWACT_DEMO_HEAD
export PYTHONPATH="$EXP:$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
run_task() (
  set -eu
  task=$1
  port=$(bash utils/get_free_port.sh)
  bash policy/KnowAct/setup_eval_policy_server.sh RoboMME "$task" knowact08-state panda joint 7 0 \
    "$KNOWACT_PYTHON" "$port" localhost >"$OUT/server-$task.log" 2>&1 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null || true' EXIT
  bash utils/wait_for_policy_server.sh localhost "$port" "$pid" KnowAct 300
  cli=("$SIM" "$EXP/probe_grasp.py" --task "$task" --port "$port" --out "$OUT" --video)
  if [[ $task == MoveCube ]]; then
    "${cli[@]}" --split train --episodes 1 4
    "${cli[@]}" --split val --episodes 1 8
    "$SIM" "$EXP/probe_grasp_feedback.py" --data "$DATA" --port "$port" --episodes 1 4 --out "$OUT/feedback"
  else
    "${cli[@]}" --split val --episodes 1 5
  fi
)
run_task MoveCube
run_task PickXtimes
"$SIM" "$EXP/probe_grasp.py" --task MoveCube --split train --expert-data "$DATA" --episodes 1 4 --out "$OUT/controls"
CUDA_VISIBLE_DEVICES=0 "$KNOWACT_PYTHON" "$EXP/probe_history.py" --task MoveCube --weights "$KNOWACT_CKPT_PATH" \
  --data "$DATA" --episodes 1 4 --out "$OUT/offline"
for kind in codec expert_history predicted_history; do
  "$SIM" "$EXP/probe_grasp.py" --task MoveCube --split train --predictions "$OUT/offline" \
    --replay-kind "$kind" --episodes 1 4 --out "$OUT/controls"
done
```
