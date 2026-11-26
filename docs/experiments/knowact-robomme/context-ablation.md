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
