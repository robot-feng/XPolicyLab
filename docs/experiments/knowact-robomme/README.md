# KnowAct V7.1 在 RoboMME 上的四卡训练

[示范关节实验：同种子成功率、原模型与动作头录像、可复现命令](context-ablation.md)

PatternLock 后续完成了单轨迹与单任务全部 100 条示范的四卡拟合，附原生成功录像、逐回合成绩以及全 16 任务输入检查。[逐级拟合与闭环实验](fit-ladder.md)

针对原训练的 0/12 结果，后续 PatternLock 单任务实验修正了示范上下文和动作时序。[修复、独立训练和验证命令](patternlock-pilot.md)

这次实验用官方 Qwen3.5-0.8B 初始化 KnowAct 的语言和视觉主干，用已经适配 Panda 的 ActionCodec 表示动作，再用四项 RoboMME 任务训练动作预测。目标为 30,000 次优化器更新。固定留出集的 CE 在第 4、256、512 步分别为 14.287、1.549、1.485。操作效果由原生仿真回合的终止状态衡量。[验收记录](preflight.json)

## 正式录像与零成功率核查

在官方 test/0、2、3 上，最终 30,000 步与离线最优 6,912 步的权重各为 0/12。24 段原始录像都是每秒 20 帧，帧数均等于原生控制步数加一帧初始观测。前视相机在左，腕部相机在右。[正式回合记录](progress-knowact08-state-30000.json)

PatternLock 的简单回合（test/0，种子 650000）要求按开场视频重画路径。前视画面中，示范的第一段沿同一行向左，模型从同一起点先向上走，第 45 个控制步收到原生 `fail`。6.6 秒的对照片前半段是在同一种子下重新取得的 86 帧开场示范，后半段取自原始评估录像。两段均为原速。[播放示范与 30K 模型执行对照](patternlock-test-0.mp4)

训练缓存从示范结束处开始截取轨迹。评估时，环境客户端在 reset 后发送示范帧；策略的 `update_obs_batch` 每次用新观测替换缓存。执行阶段第一帧到来后，`infer_stream` 才创建原生上下文，之前的示范未进入模型。由此可以确定策略缺少任务指定的示范信息。其对各任务成功率的影响仍需同种子的对照实验量化。[训练切分](../g05-robomme/prepare.py) · [环境回放](../../../benchmarks/robomme/client.py) · [观测缓存](../../../policy/G05/model.py) · [KnowAct 推理](../../../policy/KnowAct/model.py)

为检查环境动作映射与判分，回放官方 train 分割中用于离线留出的第 0 号轨迹。将官方记录的 8 维关节动作通过 `unpack_robot_state` 和同一 `Episode.take_action` 逐步送入模拟器，直接读取原生终止结果。这个对照绕过了神经网络、模型动作解码和 WebSocket。

| 任务 | 种子 | 首次成功时的控制步 |
|---|---:|---:|
| MoveCube | 14000 | 186 |
| PatternLock | 15001 | 42 |
| RouteStick | 16000 | 94 |
| InsertPeg | 13000 | 226 |

## 30K 之后是否值得续训

建议保留第一阶段的 Panda ActionCodec 和当前 30K 策略权重作为起点，先修复开场示范输入和动作标签时序，再做第二阶段的单任务训练。下面三组检查复用了原权重。正式 test 结果仍为两个断点各 0/12。[逐任务测量](action-audit.json)

**观测对应哪条动作。** 在上表四条轨迹上，对每个控制步分别读取调用 `Episode.take_action(action[t])` 前后的七维关节状态，与缓存的 `state[t]` 比较。各任务前 20 步的平均绝对误差在执行前为 0.00269–0.00521 rad，执行后均低于 0.00000002 rad。再取前 40 步中缓存有图像的 10 个时刻，两个相机均更接近执行后画面。腕部 RGB 的平均像素误差在执行前为 6.50–12.90，执行后均小于 0.002（像素范围 0–255）。当前窗口却把这份观测与从 `action[t]` 开始的 32 步配对。修正时，应从 `action[t+1]` 开始监督，历史中的已执行动作也按这一约定取值。回合末尾按有效长度截断或补齐。第一个动作需要单独配对真实的执行初始观测。[当前窗口实现](../g05-robomme/data.py) · [多轮样本构造](train.py)

**编解码器能否表达成功轨迹。** 对同样四条轨迹，把连续 32 步专家动作经实际的归一化、ActionCodec 编码、部署解码和反归一化，每段取前 16 步回放。原生环境四条均成功。MoveCube、RouteStick、InsertPeg 与原始专家动作在同一步成功，PatternLock 从第 42 步变为第 44 步。这项对照使用了专家动作，只检验动作表示和执行链路。

**策略自身能预测多准。** 使用官方 train 中每任务预留的第 0、11、22、33 条轨迹。在每条轨迹的前、中、后各取一个有 RGB 且至少有 16 步未来动作的时刻，共 48 个窗口。每次重置历史和采样种子 7，调用策略的 `update_obs` 和 `get_action`。与从下一步开始的 16 条专家指令比较，指标为七个手臂关节的平均绝对误差。相同窗口中，离线和部署预处理后的两路相机张量、本体状态、观测前缀 token 和类型逐项一致。

| 预测方式 | 48 个窗口的关节误差（rad） |
|---|---:|
| 6,912 步策略 | 0.05359 |
| 30,000 步策略 | 0.04543 |
| 在接下来 16 步重复当前关节位置 | 0.04690 |
| 给编解码器输入专家未来动作，再解码 | 0.00564 |

30K 在 PatternLock 上为 0.02424 rad，优于重复当前关节位置的 0.04051。在 InsertPeg 上为 0.05053，高于同一基线的 0.03484。这些单窗口结果可用于排查动作预测质量，正式成功率仍需由整回合评估。

下一轮建议先跑 PatternLock。训练和推理都按同一规则保留有序的开场示范，仅把它作为观测。在开跑前，用一条轨迹核对首个动作请求的输入确实含示范帧，并核对首尾动作标签及多轮历史。小规模训练后，固定多个官方 val 种子，分别输入正确示范、其他回合的示范、空示范，记录第一段路径方向与原生成功率。看到模型随示范改变动作、在多个独立回合成功，再扩大到四任务 30K。

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
bash policy/KnowAct/train.sh RoboMME knowact08-state panda joint 7 0,1,2,3 --stop-at 4 --legacy-execution-only
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
