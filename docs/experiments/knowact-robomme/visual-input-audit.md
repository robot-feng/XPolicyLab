# KnowAct 在 PatternLock 上的视觉输入与动作复核

## 送入预训练视觉编码器的数据

Qwen3.5-0.8B 的图像处理器把一张静态 RGB 图复制成两个相同的时间帧，再拆成图像块。KnowAct 导入的 G0.5 实现先创建了形状为 `[图像数, 颜色, 时间, 高, 宽]` 的张量，接着把它当作 `[图像数, 时间, 颜色, 高, 宽]` 做 reshape。以单个像素为例，视觉卷积应读到 `R,R,G,G,B,B`，旧路径给出 `R,G,R,B,G,B`。这发生在解码和归一化之后，训练和部署共享该处代码。单纯对比训练与部署的输出无法发现两端共同的偏差。[官方预处理配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/preprocessor_config.json) · [Transformers 4.57.1 参考实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py#L224-L265)

在官方 train/0 的 4 个示范时刻各取前视、腕部两张图。给两条路径完全相同的已归一化像素，在 KnowAct 实际视觉编码器入口抓取张量，以本机 Transformers 4.57.1 的 `Qwen2VLImageProcessorFast` 为参照。共比较 3,145,728 个标量，旧实现有 1,928,158 个不同，平均绝对差为 0.26497；修正后的最大绝对差为 0。加载该策略冻结的真实视觉权重，两种排列输出的视觉 token 平均余弦相似度为 0.8871。这组数只衡量视觉输入和特征变化。[可复现探针](probe_image_patches.py)

修复位于 KnowAct 的视觉模块。新的官方 Qwen 基模初始化将 `model.model_arch.vision.hf_image_patch_layout` 写入运行配置，训练和部署按该配置读取。历史权重缺少此键时保留原排列；从旧权重另开训练时可以显式使用 `--hf-image-patches`。本次实验沿用 Panda/joint、原动作编解码器、原始 RGB 通路和原生任务判定。[模型代码](../../../policy/KnowAct/G05/src/g05/models/g05/g05_model_qwen35.py) · [训练设置](train.py)

## 固定场景后，改变示范能否改变目标

用修改前的 `pattern-full100` 权重，在已有目标记录的验证回合中，按布局相同、起始按钮相同、第二个按钮不同配对，得到 (1,20)、(3,15)、(12,16)、(13,21)。每对互换供体，在两个方向各运行原示范、换示范、原示范重复三种条件，共 24 次。每次新建原生环境、重置策略，核对整个按钮布局；同一接收场景固定初始执行画面的实际像素、状态、指令和时间标记。仅替换模型采样的 4 帧示范双相机 RGB。观察到第二个不同按钮、自然终止或执行满 96 步即停止。[逐次结果](target-choice-val.csv) · [模拟器与策略服务探针](probe_target_choice.py)

| 条件 | 首段到达所示目标 | 96 步内没有第二次不同触碰 | 首段 16 步关节指令与原示范的平均差 |
| --- | ---: | ---: | ---: |
| 原示范 | 1/8 | 2/8 | 基准 |
| 只换示范画面 | 1/8 | 2/8 | 0.02885 rad |
| 原示范重复 | 1/8 | 2/8 | 0 rad |

8 个接收场景在换示范时首批指令全部发生变化，重复原示范时全部复现。val/12 的原示范要求 7 → 3，实际触碰 7 → 4；换成要求 7 → 6 的示范后，实际触碰 7 → 3。另有场景沿着大体正确的方向移动，最终仍触碰了其他按钮。当前权重会响应示范，但这些干预不能把错误完全分解为目标识别和运动控制两个比例。这里按第一段判读，完整任务成功率单独计算。

## 核实漏按目标时的物理距离

原生 PatternLock 以末端离按钮的水平距离不超过 10 毫米、末端高度低于 100 毫米判定到达。检查上轮两个只替换第 17–32 步就会失败的训练回合。以下距离来自原生轨迹记录的逐步末端位置及对应按钮中心。[官方几何判定](https://github.com/RoboMME/robomme_benchmark/blob/582ab30fa95ce5503261e03dbc7d4f37937dc39e/src/robomme/robomme_env/utils/subgoal_evaluate_func.py#L548-L570) · [原生回放及触碰记录](failure-audit-replays.csv)

| 回合、第二个按钮 | 原专家最近距离 | 专家动作经编解码后 | 只换一段模型动作 | 该处末端高度 | 第 32 步从实际状态由专家重新规划 |
| --- | ---: | ---: | ---: | ---: | --- |
| train/0、按钮 4 | 1.35 mm | 3.20 mm | 13.43 mm | 60.9 mm | 第 58 步成功 |
| train/2、按钮 3 | 1.48 mm | 0.66 mm | 11.87 mm | 74.1 mm | 第 132 步成功 |

两次都在高度范围内，水平距离分别超出判定半径 3.43、1.87 毫米。第 32 步之后继续执行原来的专家命令，两次都按错；回放相同前缀、核对关节到达相同状态后，让专家根据当时的位置重新规划，两次均完成全程。专家可读目标标签，这两次成功只检验该状态是否仍可恢复。[接管与状态一致性核对](probe_recovery.py)

## 按相同验证种子检验修正效果

原权重为 `RoboMME-knowact-pattern-full100-panda-joint-7/last.pt`，SHA-256 为 `41aa0672ef58a2b663e3f4c0c211dbc4355d7abf3adc6e4b2f659fad90aaf6c4`。冻结权重的两组在同一张卡上依次评测，仅改变视觉排列。两组短程续训都从这一权重出发，各用单卡、batch 1、种子 7、相同顺序的 96 个训练窗口和优化器设置。视觉主干冻结。四种条件均用标准部署入口、正确示范、本回合实际执行历史、16 步重规划和 1300 步预算，评测同一批 val/8–15。首段正确要求原生触碰顺序的前两项均正确。[逐回合状态](visual-layout-val.csv)

| 条件 | 完整成功 | 首段正确 | 超时回合 | 固定训练窗口 CE |
| --- | ---: | ---: | ---: | ---: |
| 原权重，旧排列 | 0/8 | 0/8 | 0 | — |
| 原权重，只切换到官方排列 | 0/8 | 2/8 | 2 | — |
| 续训 96 步，旧排列 | 0/8 | 3/8 | 1 | 0.17230 |
| 续训 96 步，官方排列 | 0/8 | 0/8 | 0 | 0.21898 |

原权重只切换输入排列后，首段从 0/8 到 2/8，两个回合超时，整回合仍为 0/8。训练窗口 CE 在相同的 32 个训练样本上计算。两个续训组都从已经适应旧输入的策略出发。旧排列续训后有 3 个回合走对第一段，官方排列续训后为 0 个，两组都没有完成整回合。诊断支持修正新的 Qwen 初始化和完整保留旧基线。目前没有证据把旧权重直接切换或短续训后的模型作为成功率更高的替代版本。

下一项区分度较高的实验，是从采用官方排列的干净 Qwen 初始化出发，在同布局、同起点、不同第二目标的少量成对训练示范上过拟合，用互换示范后的真实首段触碰验收。若目标能随示范切换却仍擦过按钮，再加入从实际偏离状态重新规划的训练监督。两项都能用原生策略单独验证。

## 复现

按 [PatternLock 环境准备](patternlock-pilot.md) 设置 `EXP`、`PILOT`、`KNOWACT_PYTHON`、`ROBOMME_PYTHON`、`KNOWACT_INITIAL`、`KNOWACT_OUTPUT_ROOT` 和 `KNOWACT_ROBOMME_DATA`，从 checkout 根目录执行。每次评估指定一个新的输出目录。

```bash
export PYTHONPATH="$EXP:$PWD/policy/KnowAct/G05/src:$PWD/policy/KnowAct/G05:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
BASE="$KNOWACT_OUTPUT_ROOT/RoboMME-knowact-pattern-full100-panda-joint-7/last.pt"
D="$PILOT/visual-input-reproduction"
mkdir -p "$D"
unset KNOWACT_DEMO_HEAD
"$KNOWACT_PYTHON" "$EXP/probe_image_patches.py" --weights "$BASE" \
  --images "$KNOWACT_ROBOMME_DATA/data/PatternLock/0/demo.npy" --output "$D/patches.json"

for layout in legacy hf; do
  flag=()
  if [[ "$layout" == hf ]]; then flag+=(--hf-image-patches); fi
  name="pattern-vision-$layout-96"
  KNOWACT_BATCH_SIZE=1 KNOWACT_MAX_STEPS=96 bash policy/KnowAct/train.sh \
    RoboMME "$name" panda joint 7 2 --task PatternLock --all-train-demos \
    --warm-start "$BASE" --save-every 96 --val-windows 32 "${flag[@]}"
  EVAL_ENV_TYPE=sim KNOWACT_KEEP_HISTORY=true KNOWACT_CONTEXT_MAX_TOKENS=65536 \
    KNOWACT_CKPT_PATH="$KNOWACT_OUTPUT_ROOT/RoboMME-$name-panda-joint-7/best.pt" \
    KNOWACT_REPLAN_STEPS=16 ROBOMME_ACTION_STEPS=16 ROBOMME_SPLIT=val \
    ROBOMME_START_EPISODE=8 ROBOMME_EPISODES=8 ROBOMME_MAX_STEPS=1300 \
    ROBOMME_OUTPUT_DIR="$D/$layout" bash policy/KnowAct/eval.sh \
    RoboMME PatternLock "$name" panda joint 7 2 2 "$KNOWACT_PYTHON" "$(dirname "$(dirname "$ROBOMME_PYTHON")")"
done
```

冻结权重的对照复用上述 eval 命令。旧排列指向 `$BASE`；新排列在单独的 run 目录硬链接同一权重、链接同一份 `dataset_stats.json`，复制 `.hydra/config.yaml` 并仅将 `model.model_arch.vision.hf_image_patch_layout` 设为 `true`。示范替换实验先用 `setup_eval_policy_server.sh` 启动原权重，再在模拟器环境运行 `probe_target_choice.py --port "$PORT" --out "$D/pairs" --pairs 1 20 3 15 12 16 13 21`。物理接管实验使用[前一轮复现命令](action-failure-audit.md)生成的 `grafts/train-{0,2}-replace_16.json`，逐条运行 `probe_recovery.py --trace <trace.json> --cutoff 32 --out <新目录>`。
