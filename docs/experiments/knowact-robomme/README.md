# KnowAct V7.1 在 RoboMME 上的四卡训练

这次实验用官方 Qwen3.5-0.8B 初始化 KnowAct 的语言和视觉主干，用已经适配 Panda 的 ActionCodec 表示动作，再用四项 RoboMME 任务训练动作预测。目标为 30,000 次优化器更新。固定留出集的 CE 在第 4、256、512 步分别为 14.287、1.549、1.485。操作效果由原生仿真回合的终止状态衡量。[验收记录](preflight.json)

## 实验条件

| 项目 | 本次设置 |
|---|---|
| 模型源 | 用户提供的 KnowAct V7.1，ZIP SHA256 为 `b190794523404b89d6fe6266c29194db58dd6ac67f9aa5565d5a06995fb8a241`，原始导入提交为 `0fbf23a9` |
| 正式基模 | `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`，语言宽度 1024，24 层，视觉编码器 12 层 |
| 备选基模 | `--backbone g05` 使用官方 G05 RoboDojo 训练基础包，语言宽度 2048，视觉编码器 24 层 |
| 数据 | 官方 train 分割的 MoveCube、PatternLock、RouteStick、InsertPeg，各 100 条完整轨迹，各用 90 条训练、10 条离线留出 |
| 观测与控制 | 20 Hz Panda，正面与腕部 RGB，7 维手臂关节加 1 维夹爪。RGB 缓存间隔 4 个原生控制步 |
| 历史与目标 | 每个样本随机请求 1–4 轮，轨迹起点处按已有历史取样。相邻轮次相距 16 个控制步；历史只输入实际执行的前 16 步，其余位置按末值补齐。当前轮次预测 32 步，BAR 块宽 16。训练使用原生 CE |
| 参数与优化器 | 训练末 4 个语言层、最终归一化层、共享输入嵌入与输出头、机器人状态投影，共 342,333,408 个参数。视觉主干冻结。AdamW，FP32 可训练参数，BF16 前向，梯度范数裁剪 1，100 步预热，峰值学习率 2e-5，余弦下降到 2e-6 |
| 并行 | 4 张 RTX 4090，每卡 1 个样本，每步同步 4 个样本。各 rank 使用相同步数、不同抽样种子。每 256 步对固定 32 个多轮留出样本计算平均 CE |
| 部署 | 每次下发前 16 步，在下一次规划时传入上一段实际执行步数和计划编号。单实例及批量 RPC 均可用 |

训练沿用四任务执行段缓存。示范视频前缀由环境客户端回放，当前策略只在动作请求时提交上下文，因此模型记录控制阶段的同回合历史。跨回合示范学习未在本次实验中训练。正式对照还涉及骨干大小、训练目标、每卡 batch 和动作生成方式的差别，单独衡量历史贡献需要同基模的历史消融实验。[G05 数据划分、编解码器和对照条件](../g05-robomme/README.md) · [官方 Qwen 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)

## 安装与数据

以下命令从 XPolicyLab checkout 根目录执行。父工作区需要已登记的 Panda 配置和 RoboMME 模拟器。策略端与模拟器分别使用各自的 Python。KnowAct 的 `g05` 源码和原 G05 同名，分别启动独立进程。

```bash
export DATA=/data/haichao-g05-robomme
export ART=/data/haichao-knowact-robomme
export EXP="$PWD/docs/experiments/knowact-robomme"
export KNOWACT_PYTHON="$PWD/policy/KnowAct/.venv/bin/python"
export KNOWACT_ROBOMME_DATA="$DATA" KNOWACT_OUTPUT_ROOT="$ART/checkpoints"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base_py="$PWD/policy/G05/G05/.venv/bin/python"
# 已有 DATA/fla6 时可直接运行 install.sh
uv pip install --python "$base_py" --no-deps --target "$DATA/fla6" fla-core==0.6.0
bash policy/KnowAct/install.sh --reuse-g05 "$base_py" "$DATA/fla6"
```

实测版本为 Python 3.10.20、PyTorch 2.7.1+cu128、Transformers 4.57.1 和 fla-core 0.6.0。安装脚本在独立的 KnowAct 虚拟环境中优先导入其源码，并复用已安装的第三方依赖。独立环境依赖这里列出的 G05 环境和 FLA 安装目录的可用性。

RoboMME 读取官方 HDF5 中的 RGB 数组并缓存为 NumPy 文件。已有同划分的 G05 数据和第一阶段产物时可直接复用。新机器先按 G05 文档获取四个 HDF5、官方基础包及 `env_metadata`，再运行以下命令。原发布包的 RoboDojo 路径另行使用准备好的 LeRobot v3.0 导出，其字段和转换约定见发布手册。[G05 准备步骤](../g05-robomme/README.md) · [KnowAct 发布手册](../../../policy/KnowAct/docs/06_复现手册.md)

```bash
bash policy/KnowAct/process_data.sh ../datasets/robomme_data_h5 \
  ../robomme_benchmark/src/robomme/env_metadata "$DATA/data"
# 只在需要重建第一阶段时，在新的输出位置按 G05 文档运行 codec.py
test -s "$DATA/run/action_tokenizer.pt"
test -s "$DATA/run/dataset_stats.json"
```

第一阶段从官方动作编解码器继续训练 1,024 步。在固定的 384 个留出窗口上，第 256 步关节重建 MAE 最好，为 0.005618 rad。本次两种初始化实际加载的编解码器参数和缓冲区均与这个断点逐张量一致，共 208 个。动作编解码器独立加载，迁移策略服务时需要同时保留该文件、归一化统计和视觉语言处理器目录。[编解码器原始曲线](../g05-robomme/result-codec.json)

## 选择基模与复现训练

下载 Qwen 后可启用离线模式。初始化输出参数要指向新目录。权重目录中的 `.hydra/config.yaml` 保存本机资产路径，跨机器部署时需更新这些路径。

```bash
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir "$ART/qwen35-08b"
export HF_HUB_OFFLINE=1
export KNOWACT_INITIAL="$ART/initial-qwen-state"
"$KNOWACT_PYTHON" "$EXP/initialize.py" --backbone qwen35-08b \
  --qwen-snapshot "$ART/qwen35-08b" --data "$DATA" --output "$KNOWACT_INITIAL"

# 选择 G05 时使用单独的输出路径和训练 run 名称
"$KNOWACT_PYTHON" "$EXP/initialize.py" --backbone g05 --data "$DATA" --output "$ART/initial-g05"
```

Qwen 初始化时，将全部 320 个唯一语言参数和 153 个视觉参数与官方 safetensors 作精确比较，检查已有嵌入行和共享 LM head。官方 MTP 辅助头未用于 KnowAct。新增动作词、机器人状态投影需要训练。原版状态投影末层的权重和偏置均为零，六个真实样本中该层出现约 10¹² 的梯度范数。Qwen 路径保留零权重，仅用已有的 `state` 词向量（ID 2374）初始化偏置。相同六个样本上，修正后最大单层梯度范数为 59–103，四卡首步全局梯度范数为 128.52。G05 路径保留官方 checkpoint loader 的迁移规则，包括词表末尾裁去 4 行、机器人投影由 27 维适配到内部 20 维。已回读其 618 个语言和视觉权重条目与源文件对齐，G05 备选路径本次只验收到加载阶段。

```bash
# 新 run 的四步验收。steps 仍是 30000，因此后续续训与预热、余弦相位连续。
bash policy/KnowAct/train.sh RoboMME knowact08-state panda joint 7 0,1,2,3 --stop-at 4
# 从最近一次完整保存继续，10K / 20K / 30K 分段训练、评估并同步报告
"$KNOWACT_PYTHON" -u "$EXP/milestones.py" --name knowact08-state \
  --sync-branch feat/g05-robomme-training
```

单段训练使用同一 `train.sh` 加 `--resume --stop-at 10000`。本机已启动的进程与调度器在 `runs/robomme/knowact/{formal,milestones}.pid` 留有记录，首段日志在 `formal.log`，后续段在 `milestones.log`。调度器独占 run 锁，在训练进程退出后占用 GPU 2 做仿真，然后释放 GPU 2 供四卡续训。默认在 10K、20K 用 MoveCube 官方 val/0，在 30K 用四任务各自官方 test/0、2、3，每次最多 1,300 步。30K 同时比较离线验证最优与最终断点，两者相同则复用结果。长回合显式允许累计 65,536 个上下文 token；训练和短验收使用发布包的 16,384 上限。评测错误保留日志和错误状态，成功数只按原生环境结果计算。

## 性能、断点与验收

训练沿用已在 G05 上验证的 FLA GatedDeltaNet、DDP gradient bucket view、25 MiB 梯度桶、fused AdamW 和按轨迹 mmap 的 RGB 缓存。启动时验证实际 FLA 内核，并在首轮反传检查所有选中的参数收到梯度。视觉注意力使用 PyTorch SDPA。单样本的历史长度变化会造成不同 rank 工作量不均。第 50–250 步的九段计时中位数为 8.03 个样本/秒，范围 7.69–8.55（约 2 次四卡更新/秒）。另一次连续 31 秒的采样中，GPU 0–3 的平均利用率分别为 55%、69%、73%、74%。尚无同配置关闭加速的配对速度或成功率测量。[预检与实测](preflight.json) · [G05 的独立加速实验](../g05-robomme/throughput.md)

`$KNOWACT_OUTPUT_ROOT/RoboMME-knowact08-state-panda-joint-7/` 保存 `last.pt`（权重、优化器、步数和计划）、`best.pt`（按离线 CE 选取的权重）、`metrics.jsonl` 和 `checkpoints/step_*.pt`（固定里程碑）。单个可恢复断点约 6.3 GiB。通过临时文件与原子替换发布。显式指定 `KNOWACT_CKPT_PATH` 可加载数据盘权重，标准路径也可以直接用 run 名解析。

```bash
export KNOWACT_CKPT_PATH="$KNOWACT_OUTPUT_ROOT/RoboMME-knowact08-state-panda-joint-7/best.pt"
EVAL_ENV_TYPE=sim ROBOMME_SPLIT=val ROBOMME_EPISODES=1 ROBOMME_MAX_STEPS=18 \
  ROBOMME_OUTPUT_DIR="$PWD/runs/robomme/knowact/smoke-new" \
  bash policy/KnowAct/eval.sh RoboMME MoveCube knowact08-state panda joint 7 0 1 \
    "$PWD/policy/KnowAct/.venv" "$PWD/../robomme_benchmark/.venv"
```

验收使用独立输出目录。真实 MoveCube val/0 第 4 步模型执行 19 个控制步，跨过一次 16 步重规划，原生终止状态为 timeout。通用 debug 客户端分别发送原始 RGB 和编码 JPEG，各完成一个 20 步回合。发布包 73 项回归和仓库 32 项回归全部通过。下面两条命令可复跑源码测试。

```bash
XPL_ROOT="$PWD" KNOWACT_ROOT="$PWD/policy/KnowAct/G05" G05_TEST_STANDALONE=0 \
  PYTHONPATH="$PWD/policy/KnowAct/G05/tests:$PWD/.." CUDA_VISIBLE_DEVICES=0 \
  "$KNOWACT_PYTHON" -m unittest discover -s policy/KnowAct/G05/tests -p 'test_*.py'
PYTHONPATH="$PWD:$PWD/.." "$KNOWACT_PYTHON" -m unittest discover -s tests
```
