# KnowAct 能否从同一起点按示范选择不同目标

在 RoboMME PatternLock 的两个固定物理起点，用同一策略、同一初始观测分别执行两次。两次唯一换给模型的是示范的四帧双相机 RGB。两次都按到各自指定的第二个按钮，这个起点才算通过。模型执行后的观测和后续动作都可以随第一次决策而改变。[诊断入口](probe_target_choice.py)

## 四条训练轨迹的结果

| 预先选定的示范 | 示范 A / B 前两步 | 初始权重实际第二触碰 A / B | 512 步后实际第二触碰 A / B | 两条都按对 |
| --- | --- | --- | --- | --- |
| train/28、45 | 0→3 / 0→4 | 无 / 无 | 3 / 3 | 否 |
| train/5、49 | 6→3 / 6→4 | 无 / 无 | 3 / 4 | 是 |
| val/1、20 | 0→4 / 0→3 | 5 / 5 | 4 / 3 | 是 |
| val/5、13 | 6→4 / 6→3 | 无 / 4 | 3 / 4 | 否 |

“无”表示截至诊断停止时尚未碰到另一个按钮。分别以 A、B 回合为接收方，以上四行在反向互换时得到相同的按钮结果。训练前，训练和验证各有 0/2 个物理起点通过；训练后，各有 1/2 个通过。同一起点的两次互换共用布局和初始状态，分别计作两个独立样本会夸大样本量。[48 次自主闭环记录](paired-learning-trials.csv)

验证起点 0 的两条完整示范为 `0→4→2→1` 和 `0→3`，都没有出现在这四条训练轨迹中。在同一初始画面，模型分别于第 20 和 25 步按到了 4 和 3。验证起点 6 的两条路线为 `6→4→1` 和 `6→3→1`，终点相同。它们都使动作发生变化，但第一条碰到 3，第二条碰到 4。两次与要求的第二按钮最近的水平距离分别为 47.2 和 50.9 毫米，明显超出 10 毫米判定半径。另有一种失败更接近控制精度问题：在 train/45 上，目标是 4，最近距离为 10.7 毫米，之后碰到 3。[起点 0 的同步录像](paired-val-start-0.mp4) · [起点 6 的同步录像](paired-val-start-6.mp4)

两段录像各固定一个接收回合。左右两栏是两次独立重置，每栏含前视和腕部相机。开头 1.6 秒依次显示真正输入模型的四个示范时刻，之后按 20 Hz 播放自主执行。较早结束的一栏停在末帧。标题中的按钮编号均为事后标注。

另在上述 8 条原示范上从头运行到 RoboMME 的自然终止，控制上限 1300 步。训练回合 3/4 完成，验证回合 1/4 完成。val/1 首段按对 4，随后要求按 2，实际按到 3；val/20 的完整路线只有 0→3，模型完成。没有在这里使用官方 test 分割或估计整个任务的成功率。[原生完整回合结果](paired-learning-full.csv)

## 输入、训练与计分的核对

- 初始策略使用官方 Qwen3.5-0.8B 语言和视觉权重、已经训练的 Panda 动作编解码器，采用官方静态图像分块。从相同权重出发，仅拟合官方 train/28、45、5、49 的 62 个窗口。4 张 RTX 4090，单卡 batch 1，全局 batch 4，随机种子 7，峰值学习率 0.0001，512 次更新。四帧等间隔示范，正面和腕部相机，训练最后四层语言块、绑定的词嵌入和输出权重、最终语言归一化及本体状态投影，冻结视觉主干和动作编解码器。[训练入口](train.py)
- 在第 512 步做唯一一次保存和离线验证。第 1 和 500 步抽到的训练 batch 的动作 token 交叉熵分别为 14.2063 和 0.01345，固定 8 个离线留出窗口在第 512 步的平均值为 3.60476。离线留出来自官方 train 内部；表中的 val 使用官方 val 种子。部署权重 SHA-256 为 `f2697fa2c0c7c3734a54ad41cd3ee4574904bcfaa8a6a93433dca0f1f6be2d6e`。
- 选择样本只用了原生布局与示范路线。每对共享整张按钮布局和起始按钮。对这八个回合，两个起点分别只有一份执行首帧双相机 RGB 哈希，初始机械臂七维关节值和末端位置在同一起点完全相等。四条训练和四条验证示范的 RGB 哈希全部不同；每条训练示范也与实际训练缓存的四帧逐字节相同。每个接收回合复用相同初始画面、关节、指令和示范时间标记。
- 策略只用开场示范 RGB、当前 RGB 和当前本体状态作出首个决策，此后每 16 步根据真实新观测及执行历史重规划。未启用示范关节输入或示范跟踪动作头。所有动作来自 KnowAct 标准 AR 服务。原生按钮路径和触碰列表仅供环境检查、计分和录像标注。
- 每个接收回合依次运行原示范、换示范和重复原示范。首段诊断在第二次不同触碰、自然终止或 96 步时停止。训练前后共 16 组原样重复，16 组的整段关节命令与触碰记录均复现。换示范后的物理触碰按所示目标判读；RoboMME 自身的任务目标始终保留接收回合的原路径，因此完整任务成绩只使用原示范。

## 复现

在 XPolicyLab 根目录按 [PatternLock 环境准备](patternlock-pilot.md) 设置 `KNOWACT_INITIAL`、`KNOWACT_ROBOMME_DATA`、`KNOWACT_OUTPUT_ROOT`、`KNOWACT_PYTHON` 和 `ROBOMME_PYTHON`。初始权重使用导入官方 Qwen 的初始化目录。实验脚本位于 checkout 内，模拟器使用父工作空间的 Panda 配置。

```bash
EXP="$PWD/docs/experiments/knowact-robomme"
export PYTHONPATH="$EXP:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
export KNOWACT_BATCH_SIZE=1 KNOWACT_MAX_STEPS=512
NAME=knowact-paired-hf512
bash policy/KnowAct/train.sh RoboMME "$NAME" panda joint 7 0,1,2,3 \
  --task PatternLock --train-episode 28 45 5 49 --hf-image-patches \
  --lr 0.0001 --save-every 512 --val-windows 8
```

训练前的对照在初始化目录旁创建单独服务目录，使用同一份模型权重和归一化参数。下面的配置让该对照与训练后策略采用相同的视觉排列与四帧示范。

```bash
export CLEAN="$(dirname "$KNOWACT_INITIAL")/paired-clean-base"
"$KNOWACT_PYTHON" - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
src, dst = map(Path, (os.environ["KNOWACT_INITIAL"], os.environ["CLEAN"]))
(dst / ".hydra").mkdir(parents=True, exist_ok=True)
for name in ("model_state_dict.pt", "dataset_stats.json"):
    if not (dst / name).exists():
        if name.endswith(".pt"):
            os.link(src / name, dst / name)
        else:
            (dst / name).symlink_to((src / name).resolve())
    assert (src / name).samefile(dst / name)
cfg = OmegaConf.load(src / ".hydra/config.yaml")
arch = cfg.model.model_arch
arch.vision.hf_image_patch_layout = True
arch.native_demo_frames = 4
arch.native_demo_proprio = False
cfg.run_dir = str(dst)
OmegaConf.save(cfg, dst / ".hydra/config.yaml", resolve=True)
PY
```

每种权重各启动一次标准策略服务，再在同一 checkout 运行探针。下面使用训练后的权重；对照组将 `KNOWACT_CKPT_PATH` 指向 `$CLEAN/model_state_dict.pt`。每种权重使用新的结果目录。同一服务的三个命令共用该目录名作为 websocket evaluation ID，文件名按 split、回合和条件区分。

```bash
export KNOWACT_NATIVE_CONTEXT=true KNOWACT_KEEP_HISTORY=true KNOWACT_REPLAN_STEPS=16
export KNOWACT_CONTEXT_MAX_TOKENS=65536
unset KNOWACT_DEMO_HEAD
export KNOWACT_CKPT_PATH="$KNOWACT_OUTPUT_ROOT/RoboMME-$NAME-panda-joint-7/best.pt"
PORT=$(bash utils/get_free_port.sh)
bash policy/KnowAct/setup_eval_policy_server.sh RoboMME PatternLock "$NAME" panda joint 7 \
  0 "$KNOWACT_PYTHON" "$PORT" localhost > "/tmp/knowact-$PORT.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
bash utils/wait_for_policy_server.sh localhost "$PORT" "$SERVER" KnowAct 300
export CUDA_VISIBLE_DEVICES=0
OUT=$(mktemp -d)
"$ROBOMME_PYTHON" "$EXP/probe_target_choice.py" --port "$PORT" --split train \
  --out "$OUT" --pairs 28 45 5 49
"$ROBOMME_PYTHON" "$EXP/probe_target_choice.py" --port "$PORT" --split val \
  --out "$OUT" --pairs 1 20 5 13 --video
"$ROBOMME_PYTHON" "$EXP/trace_patternlock.py" --port "$PORT" --split val \
  --data "$KNOWACT_ROBOMME_DATA" --out "$OUT" --episodes 1 20 5 13
```
