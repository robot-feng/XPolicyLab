# PatternLock 示范与动作时序修复

这轮实验从四任务 30K 的策略参数开始，使用新的优化器和独立输出目录。只使用 PatternLock 的官方 train 数据。沿用原划分，90 条用于训练，10 条用于离线验证。Panda 的第一阶段 ActionCodec、归一化统计、Qwen3.5-0.8B 和 16 步重规划间隔沿用原实验。

## 时间轴与输入

在每个 train 种子上重新调用官方环境的 reset。该调用返回一段示范和最后一帧执行初始观测。第一条专家动作配这帧执行观测。对于第 `t > 0` 条专家动作，新缓存从官方记录的第 `t-1` 条执行数据取状态和画面。关节按 20 Hz 保存，RGB 在执行时刻 0、4、8 等处采样。编解码器仍读取连续 32 个动作，回合末尾按最后有效动作填充并标记填充位置。相邻历史轮按 16 步取样，只有当时实际执行的部分作为历史。训练指令取该种子在当前模拟器 reset 中给出的首条指令。

开场示范按时间均匀取最多 4 帧，包含第一和最后一帧。每帧的前视和腕部画面独立使用原相机处理器。训练时这些帧作为纯视觉前缀加入同回合样本，其 token 不计算损失。服务时客户端标明示范帧序号与总长度，KnowAct 按权重目录中保存的 `native_demo_frames` 选帧。第一次动作请求按同一顺序写入上下文，以后通过该环境自己的 KV 缓存保留信息。旧权重默认选 0 帧。验证中的空示范和其他回合示范由环境客户端选择，原生判分使用目标回合自身的环境。

## 数据准备与时序验收

从 checkout 根目录运行。原缓存和已训练的权重位于数据盘，核对时策略和模拟器分别使用自己的 Python。新缓存引用原始动作数组和第一阶段资产，迁移时需保留这些依赖。

```bash
export ART=/data/haichao-knowact-robomme
export EXP="$PWD/docs/experiments/knowact-robomme"
export PILOT="$PWD/runs/robomme/knowact/patternlock-pilot"
export ROBOMME_PYTHON="$PWD/../robomme_benchmark/.venv/bin/python"
export KNOWACT_PYTHON="$PWD/policy/KnowAct/.venv/bin/python"
export KNOWACT_ROBOMME_DATA="$ART/preaction"
export KNOWACT_INITIAL="$ART/initial-qwen-state"
export KNOWACT_OUTPUT_ROOT="$PILOT/checkpoints"
export OLD="$ART/checkpoints/RoboMME-knowact08-state-panda-joint-7/checkpoints/step_030000.pt"
export NEW="$KNOWACT_OUTPUT_ROOT/RoboMME-knowact-pattern-demo-panda-joint-7"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PILOT"

CUDA_VISIBLE_DEVICES=0 bash policy/KnowAct/process_data.sh aligned \
  --source /data/haichao-g05-robomme --h5 ../datasets/robomme_data_h5 \
  --output "$KNOWACT_ROBOMME_DATA" --task PatternLock
PYTHONPATH="$PWD:$PWD/.." "$ROBOMME_PYTHON" "$EXP/verify_aligned.py" capture \
  --data "$KNOWACT_ROBOMME_DATA" --capture "$PILOT/capture.npz"
PYTHONPATH="$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/.." \
  "$KNOWACT_PYTHON" "$EXP/verify_aligned.py" check \
  --data "$KNOWACT_ROBOMME_DATA" --capture "$PILOT/capture.npz" --weights "$OLD"
```

验收程序在留出的 train/0 用专家动作跑到原生 success。它比较独立 reset 的示范像素、执行段开始的观测、训练窗口与真实专家动作，以及跨 16 步的历史。在真实权重的批量推理接口上拦截原生上下文追加，核对每个示范前缀的张量与训练路径逐项相同，再检查动作输出和回合重置。

本机已完成全部 100 个回合的缓存。train/0（种子 15001）的 48 帧示范选中第 0、15、31、47 帧。第一次动作请求向上下文追加 5 次，返回 16 条有效动作。12 个执行窗口均通过目标动作、填充及观测检查。独立专家回放的前 5 个缓存时刻，两路 RGB 平均绝对像素误差为 0.00171（范围 0–255）。留出验证和训练分别包含 267 和 2,371 个可选起点。

## 四卡训练与独立验证

```bash
KNOWACT_MAX_STEPS=512 bash policy/KnowAct/train.sh RoboMME knowact-pattern-demo panda joint 7 0,1,2,3 \
  --task PatternLock --warm-start "$OLD" --save-every 256
for mode in matched other none; do
  EVAL_ENV_TYPE=sim KNOWACT_CONTEXT_MAX_TOKENS=65536 KNOWACT_CKPT_PATH="$NEW/last.pt" \
    ROBOMME_DEMONSTRATION="$mode" ROBOMME_SPLIT=val ROBOMME_START_EPISODE=0 ROBOMME_EPISODES=4 \
    ROBOMME_MAX_STEPS=1300 ROBOMME_OUTPUT_DIR="$PILOT/val-$mode" \
    bash policy/KnowAct/eval.sh RoboMME PatternLock knowact-pattern-demo panda joint 7 0 0 \
    "$KNOWACT_PYTHON" "$PWD/../robomme_benchmark/.venv"
done
```

验证固定使用官方 val/0–3，每个条件都重新创建目标回合。其他回合示范取同分割的下一条轨迹。环境终止后记录实际触碰和目标按钮序列，可核对是否沿正确的第一段移动。每种条件的 report.json 与 MP4 录像写入单独目录。本轮开发使用 val，原四任务 test 评估留作正式比较。
