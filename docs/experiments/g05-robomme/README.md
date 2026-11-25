# G05：RoboMME 四任务两阶段训练

目标是测量 G05 在 Panda 关节控制下的任务完成率。使用 `MoveCube`、`InsertPeg`、`PatternLock`、`RouteStick` 四项本地可用的官方训练数据，不把它们的分数推广到整个 RoboMME。环境端继续使用 `benchmarks/robomme/`。

## 数据和预注册的评估方法

HDF5 来自 `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337`。每项 100 条官方 `train` 轨迹。按完整 episode 划分：训练 90 条，离线验证使用编号 `0,11,22,33,44,55,66,77,88,99` 的 10 条。两阶段共用该划分和仅由训练集计算的 1%/99% 分位数。独立仿真预先固定官方 `test` 的 `0,2,3`（容易、中等、困难）三个原生回合，每个任务分别记录成功数和原生终止状态。选模型只用离线验证集。

原始 HDF5 含示范视频前缀和执行段。本次训练当前帧基线，只从执行段生成目标。训练输入采用前视和腕部 RGB、七个关节角、第一根夹指的位置及任务文本。核心网络每四帧取一个观测，预测后续连续 32 个原生动作（七个绝对关节目标及一个夹爪命令）。末尾只在当前轨迹内补齐，核心 loss 带 padding mask。编解码器从完整 32 步窗口采样。每次转换都比对任务和 episode seed；已逐项从原始 HDF5 回读第 11 轨迹，检查两路 RGB、关节状态和动作。

在本仓库根目录设置路径，`ART` 指向数据盘，不要把权重放进 Git：

```bash
export G05="$PWD/policy/G05/G05" PY="$PWD/policy/G05/G05/.venv/bin/python"
export EXP="$PWD/docs/experiments/g05-robomme" ART=/data/haichao-g05-robomme
export PYTHONPATH="$EXP:$G05:$G05/src:$PWD/.." OMP_NUM_THREADS=2
python "$EXP/prepare.py" ../datasets/robomme_data_h5 \
  ../robomme_benchmark/src/robomme/env_metadata "$ART/data"
```

基础训练包为 `OpenGalaxea/g05-robodojo@bb6cf7e45d0b2e8bd71c16c52961a93cc6125283` 中的 `g05_robodojo_train_base_assets.tar`，SHA256 `188fc8f092aa3ed2f67fcdb4a3e94ddbb44d4e0c41309e9f672a5a8594291218`。已对整包校验，并将其 `pretrained/` 内容解至 `$ART/base/`。其中 `action_tokenizer_hf/` 是第一阶段起点，`checkpoints/model_state_dict.pt` 是核心模型起点，`hf_processor/` 是匹配的视觉语言处理器。两阶段都从官方权重开始，不使用双臂 RoboDojo 成品模型做 Panda 初始化。

## 已验证的第一阶段结果

400 条轨迹共 70,038 个执行帧。核心训练集有 360 条轨迹、63,070 帧、15,896 个带图像的观测窗口；留出集有 40 条轨迹、6,968 帧、1,755 个带图像的观测窗口。每项任务的 10 条留出轨迹覆盖 5 个容易、2 个中等、3 个困难回合；它们与官方测试 seed 无交叉。

已逐个将加载后的 208 个 ActionCodec 张量与基础 `model.safetensors` 对比，相同且无缺失。第一阶段完成 1024 次更新。在相同 384 个留出窗口上，基础模型的平均关节绝对误差是 0.005728 rad，最佳为第 256 步的 0.005618 rad，最后一步为 0.006501 rad。因此第二阶段读取第 256 步的 `action_tokenizer.pt`。误差降幅约 1.9%，这里只度量动作还原，不等于任务成功率。[各任务样本数和完整验证曲线](result-codec.json)。

## 训练、选模与仿真

1. `codec.py` 从官方 ActionCodec 启动，适配编码器与解码器，固定量化字典以保留原有 token 编号。夹爪沿用上游二值规则。训练 1024 次，batch 24，AdamW 学习率 `2e-5`；每 128 步用同一组 384 个留出窗口评估 **两层残差码** 的七关节重建 MAE（弧度），在基础权重和历次训练权重中选误差最低的一次。
2. `core.py` 使用选出的 tokenizer 和同一批训练示范，按上游联合目标训练离散动作（CE）和连续动作（FM）。每张 4090 只有 24 GiB；这里更新 VLM 和动作专家各自最后 4 层，以及本体状态与动作投影，共 343,294,964 个参数；视觉主干冻结。可训练权重保留 FP32，前向用 BF16。目标为累计 **30,000 次优化器更新**。前 1,024 步使用三卡（全局 batch 6），之后使用四卡（全局 batch 8），每卡 batch 均为 2。按此配置共随机抽取 237,952 个训练窗口，相当于现有 15,896 个窗口的 14.97 倍。每 256 步、10K、20K 和最后一步计算同一组 128 个留出窗口的联合 loss；按该指标保存 `best.pt`，并将当前参数、优化器及步数存入 `resume.pt`。恢复时保持每卡 batch，加 `--resume` 并传入目标总步数；恢复参数和 AdamW 一、二阶矩，余弦学习率从保存时的相位延长到新终点 `2e-6`，不重复预热。窗口抽样按全局步数接续；旧断点不保存数据增强与扩散噪声的全部 RNG 状态，因此不承诺逐位复现。两种 loss 量纲不同，不把其和解释成成功率。
3. 基线和微调权重均使用这次训练集的归一化统计；独立服务用相同种子启动，都取连续动作头，执行每个 32 步预测的前 16 步。每个回合交给原生环境终止和判分，预算 1300 步。协议、模拟器、动作打包和评分不为 G05 特化。

```bash
CUDA_VISIBLE_DEVICES=3 "$PY" "$EXP/codec.py" "$ART"
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=4 "$EXP/core.py" "$ART" --steps 30000
for kind in base tuned last; do
  "$PY" "$EXP/export.py" "$ART" "$kind"
  POLICY_GPU=2 ENV_GPU=2 bash "$EXP/evaluate.sh" \
    "$ART/eval-$kind/model_state_dict.pt" "$PWD/runs/robomme/g05-$kind"
done
```

`run/dataset_stats.json`、`run/action_tokenizer.pt`、`run/.hydra/config.yaml` 和选中 checkpoint 是推理所需的配套文件。导出器将增量核心参数合回官方基础权重。官方 loader 会将基础模型 27 维接口的相应投影截到任务使用的 20 维内部布局；Panda 的真实动作占其中 8 维（右臂 7 维＋夹爪 1 维），其余维度均有 mask。观察历史长度为 1，因此本轮不测试记忆增强版 G05。

已用真实样本验证两项核心 loss 的反向传播，并在单独的临时目录验证双卡优化、验证与保存。用实验分支的现有 `policy/G05/setup_eval_env_client.sh` 启动 `MoveCube/val/0`，预算 2，G05 通过 WebSocket 在真实模拟器执行 3 步并得到原生 `timeout`。这是通信与动作映射测试，不计入效果对照。32 项仓库回归通过，包括新加的单、双臂状态/动作往返及相机映射测试。

## 本机训练跟踪

正式续训使用 GPU 0、1、2、3，每卡一个 DDP 进程，目标已从 7,948 步延长到累计 30,000 步；第一阶段维持已完成的 1,024 步。查看本机进度用 `tail -f runs/robomme/g05-training/core.log`；只看固定验证集，读取 `$ART/run/core_metrics.jsonl`。`runs/robomme/g05-training/{base,tuned,last}/<任务>-<回合>/report.json` 是原生仿真记录。[基础模型的正式对照](result-base.json)为 0/12。`result-core.json` 将保存整条验证曲线；`result-tuned.json` 和 `result-last.json` 将分别记录验证最优和第 30,000 步在相同测试场景的逐任务成绩。若最优正好是最后一步，则复用同一批仿真记录。每个任务只有三个测试回合，报告样本数与终止状态。

上面的命令演示从头开始的四卡训练。本机实际在第 1,024 步的完整断点从三卡切成四卡。已核实 121 组 AdamW 状态均为 1,024 步，四卡第 1,025 步完成梯度同步和更新（loss `2.373102`）。`nvidia-smi` 逐个 PID 对照确认 GPU 0、1、2、3 各承担一个 rank；GPU 2 的训练进程当时占用 23,460 MiB。每次优化同步八个窗口的梯度；续用 AdamW 和原有学习率，不随 batch 放大学习率。离线验证保留原先的三组窗口、每组随机种子和 batch 顺序，使切换前后的 loss 可以比较。四卡训练期间不并行启动仿真。10K 和 20K 用 `--steps 30000 --stop-at 10000` / `--stop-at 20000` 保存并释放训练进程，短暂在 GPU 2 跑同一验证回合，之后用四卡和 `--resume` 继续。30K 再运行预定的正式测试。

本次实际在第 768 步切换到 30K：核实 121 组 AdamW 状态的 `step` 均为 768，再从完整断点恢复；第 769 步已完成反向传播和优化。学习率从 `1.958848913e-5` 接到 `1.958822790e-5`。固定留出窗口在 256、512、768 步的联合 loss 分别是 `2.717556`、`2.488617`、`2.405152`。还单独从真实的 `resume.pt` 导出过推理包，比对全部 121 个微调张量一致。

在第 256、10,000、20,000 步保存候选，在 `MoveCube/val/0` 与基础权重作同 seed 对照（基础模型同一回合只跑一次）。10K、20K 增量存于 `$ART/run/step-{10000,20000}.pt`；用 `export.py "$ART" tuned --step 10000` 等命令合并，输出 `eval-step-10000/`。每个导出命令要求目标目录尚不存在。`progress-10000.json`、`progress-20000.json` 记录阶段对照。这是中途观察，不参与选模，也不混入 12 个测试回合。复现单独验证回合时给 `evaluate.sh` 设置 `ROBOMME_SPLIT=val ROBOMME_TASKS=MoveCube ROBOMME_EPISODE_IDS=0` 即可。正式评估不传这些覆盖值。

第 256 步在这 128 个固定窗口上的联合验证 loss 为 `2.717556`。已把该步独立导出，并逐个对照了推理包的 946 个张量：121 个等于训练快照（343,294,964 个可训练参数），另外 825 个等于官方基础权重。本机快照位于 `$ART/preview-256/eval-tuned/`，不会随着 `run/best.pt` 更新而被覆盖。

要看模型具体做了什么，在 `runs/robomme/g05-training/{base,tuned}/` 下打开相同任务和回合目录的 MP4：左边是前视相机，右边是腕部相机。以同目录 `report.json` 的 `benchmark_seed` 核对场景，以原生 `status` 判定 `success`、`fail` 或 `timeout`；超时也计入分母。中途对照的视频在 `val256-base/` 和 `val256-tuned/`。

## 专家动作回放（环境正对照）

取每个任务第 11 条轨迹（官方 `train` 中本实验留作验证的困难轨迹），用相同 seed 重置模拟器。从数据缓存读取每个执行步的八维 `action.npy`，经 `unpack_robot_state` 和环境客户端 `Episode.take_action` 执行。四条轨迹均由原生环境判为 `success`。这验证了数据转换的时间顺序、Panda 动作打包与原生成功判定可以一起工作；下表不计入任何模型成绩。

| 任务 | seed | 达成步数 |
| --- | ---: | ---: |
| MoveCube | 15100 | 91 |
| PatternLock | 16100 | 129 |
| RouteStick | 17100 | 194 |
| InsertPeg | 14100 | 247 |

<details><summary>在本机复查（RoboMME 的 Python 环境）</summary>

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD:$PWD/.." ../robomme_benchmark/.venv/bin/python - <<'PY'
import os
from contextlib import closing
from pathlib import Path
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from XPolicyLab.benchmarks.robomme.client import Episode
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state

root = Path(os.environ["ART"]) / "data"
dims = get_robot_action_dim_info("panda")
for task in ("MoveCube", "PatternLock", "RouteStick", "InsertPeg"):
    builder = BenchmarkEnvBuilder(env_id=task, dataset="train", action_space="joint_angle",
                                  gui_render=False, max_steps=1300)
    with closing(builder.make_env_for_episode(11)) as env:
        ep = Episode(env, dims, None)
        for action in np.load(root / task / "11/action.npy"):
            ep.take_action(unpack_robot_state(action, "joint", dims))
            if ep.is_episode_end():
                break
        print(task, builder.resolve_episode(11), ep.steps, ep.info["status"])
        assert ep.terminated and ep.info["status"] == "success"
PY
```

</details>
