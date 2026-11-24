# G05、π0.5 环境配置与 RoboMME 接入记录

## 工作约定与阶段验收

本工作基于当前本地 XPolicyLab 快照 `4a9e3afd`，在
`feat/robomme-g05-pi05` 分支开发，阶段成果同步到
<https://github.com/robot-feng/XPolicyLab>。该仓库的 `main` 已有另一套重构，
因此不将本分支强推或覆盖到 `main`；原始项目远端保留为 `upstream`。

按照以下顺序推进，每一阶段保留实际执行命令、结果与限制：

| 阶段 | 内容 | 验收条件 | 当前状态 |
| --- | --- | --- | --- |
| 0 | 理解项目与主机检查 | 梳理模型/通信/评估边界；检查 GPU、现有环境及 GitHub 访问 | 已完成 |
| 1 | 原生模型测试 | 两套隔离依赖；官方权重校验；直接调用上游推理，输出非空、有限、形状正确的动作；保存机器可读记录 | 已完成，两个模型均 PASS |
| 2 | 已有 XPolicyLab 适配验证 | G05 真实权重加载；无模拟器 debug 闭环；普通 RGB 与共享编码 RGB 两条路径通过 | 待开始；π0.5 的新环境接口随阶段 3 单独验证 |
| 3 | RoboMME 环境接入 | Panda 双处注册；观测与动作转换；真实任务 reset/step/终止；自动化回归 | 待阶段 2 通过 |
| 4 | RoboMME 模型评估 | 相同任务、split、episode、动作与时间预算下运行；输出逐 episode 结果及视频 | 待阶段 3 通过 |

“依赖能 import”“原生推理通过”“通信闭环通过”“仿真任务成功”是不同验收条件。
没有执行的项目不记为通过；缩短步数的 smoke 测试不作为正式基准成绩。
每一阶段通过后提交代码、文档和可复现的验证记录，再推进下一阶段。

## 项目结构与运行边界

```text
<workspace>/
├── XPolicyLab/                     # 本仓库；作为 XPolicyLab 包被导入
│   ├── policy/G05/                 # 已有 G05 适配器
│   │   └── G05/                    # 已包含上游模型代码
│   ├── policy/Pi_05/               # 已有 π0.5 适配器
│   │   └── openpi/                 # 已包含 uv 管理的 OpenPI 代码
│   ├── setup_policy_server.py      # 从 deploy.yml + overrides 创建模型服务
│   ├── client_server/ws/           # WebSocket 协议、会话与 RPC
│   ├── utils/setup_env_client.sh   # 按 EVAL_ENV_TYPE 选择环境客户端
│   ├── utils/process_data.py       # 唯一共享图像编码/解码、状态打包、动作维度入口
│   └── tests/                     # 合同、权重解析与真实 loopback WebSocket 回归
├── env_cfg/                       # 运行时配置在父工作区，本仓库内没有此目录
└── robomme_benchmark/              # 本机已有官方 RoboMME 源码及 .venv
```

策略端通过 `XPolicyLab.policy.<目录名>.model.Model` 导入模型。客户端向独立的
策略服务发送 RGB 观测并获取动作 chunk，模型依赖与模拟器依赖可以互不混装。
现有 shell 入口统一接收 benchmark、task、checkpoint、robot、action、seed、
两个 GPU ID、策略环境和评估环境十项参数。

当前公共环境路由只有 `sim`、`debug` 和 `real_world`；`sim` 调用父工作区的
`scripts/eval_policy.sh`，不能在当前独立 checkout 中直接启动 RoboMME。
RoboMME 应新增共享环境客户端，而不是为每个策略重复实现整套模拟器接口。

必须遵守的共享约定：

- `model.py` 接收已解码 RGB，不自行解码或翻转颜色。
- 离线图像仅使用 `decode_image_bit` / `encode_image_bit`。
- 运行时动作维度通过 `get_robot_action_dim_info`，训练 shell 通过
  `utils/get_action_dim.sh`；两者读取的机器人元数据文件不同，新增机器人必须双处注册。
- checkpoint 路径通过 `utils/checkpoint_resolver.py`；缺失的显式路径不能回退到其他模型。
- 根目录中英文 README 同步，模型运行文件和数十 GB 权重不提交到 Git。

## 阶段 0：已执行的检查

日期：2026-11-24（Asia/Shanghai）。以下为本机证据，不能推断其他机器也已具备环境。

| 检查 | 实际结果 |
| --- | --- |
| `git status --short --branch` | 初始工作区干净，位于上游 `main`；本地为浅克隆 |
| GitHub 仓库检查 | `robot-feng/XPolicyLab` 存在，且与当前本地版本有显著差异 |
| GitHub SSH 认证 | 使用本机已有 robot-feng SSH identity，返回 `Hi robot-feng!` |
| `nvidia-smi` | 4 × RTX 4090，单卡 24 GB；驱动 570.133.20，CUDA 12.8 |
| 存储与内存 | 初查 `/home` 可用约 303 GB，系统内存约 251 GiB |
| 工具 | uv、Conda、Python、CUDA nvcc、ffmpeg 可用；gh CLI 未安装 |
| RoboMME 虚拟环境 | `<workspace>/robomme_benchmark/.venv/bin/python`，Python 3.11.5、Torch 2.9.1+cu128，CUDA 可用 |
| 真实 RoboMME 环境探测 | `PickXtimes/test/episode 0` 成功 reset；获得两路 256×256 RGB、7 维关节、6 维末端状态、2 维夹爪状态；执行 3 次保持动作后触发 timeout；正常 close |

RoboMME 探测使用现有环境，核心调用如下；这只是模拟器可用性探测，未加载模型：

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder
import numpy as np

builder = BenchmarkEnvBuilder(
    env_id="PickXtimes", dataset="test", action_space="joint_angle",
    gui_render=False, max_steps=2,
)
env = builder.make_env_for_episode(0)
obs, info = env.reset()
action = np.r_[np.asarray(obs["joint_state_list"][-1]).reshape(-1), 1].astype(np.float32)
for _ in range(4):
    obs, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## 模型与 RoboMME 的已知差异

现有 G05 和 Pi_05 适配默认使用 RoboDojo 的 ARX X5 双臂配置：
每臂 6 个关节加 1 个夹爪量，总共 14 维，使用头部和双腕三路相机。
RoboMME 使用 Panda 单臂，`joint_angle` 是 7 关节加 1 夹爪动作，共 8 维，
使用 front/wrist 两路相机。不能直接将双臂动作裁切后作为 Panda 模型成绩。

RoboMME 返回的观测字段是列表；reset 可能附带条件演示视频，不能悄悄丢弃全部
历史帧。其夹爪动作 `-1` 表示闭合、`+1` 表示张开，而状态为两个指关节。
末端 `ee_pose` 接口为世界坐标 xyz + XYZ Euler + gripper；XPolicyLab 标准末端
姿态为 xyz + qw/qx/qy/qz，接入时必须显式转换并验证。

官方资源：

- G05 代码：<https://github.com/OpenGalaxea/GalaxeaVLA>，当前源码已 vendored。
- G05 主权重：<https://huggingface.co/OpenGalaxea/G05>，本次下载检查发现需授权。
- G05 RoboDojo FM 权重：<https://huggingface.co/OpenGalaxea/g05-robodojo>，未设 gate，正在验证下载。
- OpenPI：<https://github.com/Physical-Intelligence/openpi>。
- RoboMME：<https://github.com/RoboMME/robomme_benchmark>。
- RoboMME 官方 π0.5 baseline：<https://huggingface.co/Yinpei/pi05_baseline>，
  对应原生代码 <https://github.com/RoboMME/robomme_policy_learning>。

尚未找到 G05 的 RoboMME/Panda 专用公开 checkpoint。该限制不会用其他机器人权重
或假动作测试掩盖；需要先完成原生模型验证，再确定可验证的接入范围。

## 本地运行产物

`runs/setup/` 保存依赖安装日志与下载日志，`policy/**/checkpoints/` 和 `.venv/`
已被忽略。阶段验收后将精简且不含凭证的验证记录保存到版本控制中的文档目录。
网络代理仅由调用时的环境变量传入，不将本机端口、凭证或 SSH 私钥写进配置。

## 阶段 1A：隔离环境与验证工具

### 安装

以下命令从 XPolicyLab 仓库根目录运行。两套环境独立，不向 Conda base 混装
Torch、JAX 或模型依赖。首次安装会下载较大的 CUDA wheel；uv 缓存可复用。

```bash
# G05：Python 3.10、Torch 2.7.1+cu128、Transformers 4.57.1
bash policy/G05/install.sh --inference

# π0.5：沿用原始 uv.lock 和安装脚本，不改模型实现
(cd policy/Pi_05 && bash install.sh)
```

G05 的 `--inference` 是仅面向推理的依赖配置：保留上游模型代码，使用原有
PyTorch SDPA / delta-rule fallback；不安装训练器、其他模拟器、FA4 等全量依赖。
因此它不是完整训练环境，不能用此配置宣称训练已验证。不传该参数时仍保留原安装方式。

π0.5 保留默认 dev dependency group：vendored OpenPI 的运行时代码有无条件
`import pytest`，删掉 dev group 会导致推理模块无法导入。

### 验证依赖与 GPU

```bash
CUDA_VISIBLE_DEVICES=0 policy/G05/G05/.venv/bin/python - <<'PY'
import torch
from g05.models.g05.inferencer import PolicyInferencer
from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
x = torch.ones((64, 64), device="cuda")
assert bool(torch.isfinite(x @ x).all())
print(torch.__version__, torch.cuda.get_device_name(0))
PY

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
policy/Pi_05/openpi/.venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp
from openpi.policies import policy_config
from openpi.training import config
assert any(device.platform == "gpu" for device in jax.devices())
assert bool(jnp.isfinite(jnp.ones((64, 64)) @ jnp.ones((64, 64))).all())
print(jax.__version__, jax.devices())
PY
```

本机结果：上述两项均通过；G05 为 `2.7.1+cu128`，π0.5 JAX 为 `0.5.3`，
均使用 RTX 4090。它们验证的是依赖和 GPU 运算，尚不等于权重推理通过。

### 校验下载

`scripts/download_pretrained_models.py` 用固定 Hugging Face revision，或 GCS
对象的固定 generation 下载官方文件。G05 校验官方 SHA-256；GCS 普通对象
校验 MD5，组合对象校验 CRC32C。文件经校验后才移除 `.partial` 后缀。
GCS 对象清单和每个文件的来源、长度、校验值写入 `download_manifest.json`。

下载工具的 Python 依赖可通过临时 uv 环境提供，不需要装进模型环境：

```bash
uv run --no-project --python 3.11 --with google-crc32c==1.9.0 \
  python scripts/download_pretrained_models.py g05 --parts 8
uv run --no-project --python 3.11 --with google-crc32c==1.9.0 \
  python scripts/download_pretrained_models.py g05-assets --parts 8
uv run --no-project --python 3.11 --with google-crc32c==1.9.0 \
  python scripts/download_pretrained_models.py pi05 --workers 3
```

G05 FM archive 约 34.43 GB，processor 所在训练资产 archive 约 13.70 GB，
π0.5 LIBERO checkpoint 约 12.44 GB。G05 主仓库 gated，本流程使用可公开读取的
RoboDojo 发布包；FM archive 本身未带 HF processor，需要从同一官方发布的训练
资产中提取匹配的 processor，不使用另一个模型的 tokenizer 顶替。

网络不稳可原命令重跑。`--parts` 开启 HTTP Range 分片，恢复时必须沿用原分片数。
已经存在但校验失败的最终文件不会被覆盖；先检查文件原因再处理。相同目标不要同时
运行多个下载命令。如果网络需要代理，请在 shell 中设置自己的 `HTTPS_PROXY` 等变量。

### 验证工具自身

```bash
git diff --check
bash -n policy/G05/install.sh
python -m py_compile scripts/download_pretrained_models.py scripts/smoke_native_model.py
policy/G05/G05/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

下载回归使用真实本地 HTTP 服务验证字节范围恢复、跨分片恢复、损坏内容拒绝、
已有文件保护、分片布局不匹配拒绝；不依赖公网或模型权重。
原生推理工具在 checkpoint 缺失时确实返回非零并保存 `FAIL` 报告，已执行验证。

阶段 1A 的留档证据：

- [G05 环境与 GPU 检查](validation/stage1a-g05-environment.json)。
- [π0.5 环境与 GPU 检查](validation/stage1a-pi05-environment.json)。
- [32 项回归测试原始输出](validation/stage1a-regression.txt)：27 项原有回归 + 5 项下载完整性回归，全部通过。

## 阶段 1B：原生权重推理验收

此阶段已完成：π0.5 与 G05 均已使用完整官方权重通过连续两次原生推理。
完整权重校验、必要 sidecar 提取完成后，用各自模型解释器运行：

```bash
# G05：先解压下载并校验后的官方 FM archive。
tar -xf policy/G05/checkpoints/g05_robodojo_sim/g05_robodojo_fm_only_checkpoint.tar \
  -C policy/G05/checkpoints/g05_robodojo_sim

# 只提取推理需要的 processor，不解压训练资产里的额外模型权重。
tar -xf policy/G05/checkpoints/g05_robodojo_train_assets/g05_robodojo_train_base_assets.tar \
  -C policy/G05/checkpoints/g05_robodojo_train_assets \
  g05_robodojo_train_base_assets/pretrained/hf_processor
export G05_HF_PROCESSOR_PATH="$PWD/policy/G05/checkpoints/g05_robodojo_train_assets/g05_robodojo_train_base_assets/pretrained/hf_processor"
CUDA_VISIBLE_DEVICES=0 policy/G05/G05/.venv/bin/python scripts/smoke_native_model.py g05 \
  --checkpoint policy/G05/checkpoints/g05_robodojo_sim/hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint \
  --output runs/native-g05-01

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
policy/Pi_05/openpi/.venv/bin/python scripts/smoke_native_model.py pi05 \
  --checkpoint policy/Pi_05/checkpoints/pi05_libero \
  --output runs/native-pi05-01
```

G05 直接调用 upstream `PolicyInferencer`，使用 FM 模式与 checkpoint 自身的
processor schema 构造输入。π0.5 直接调用 OpenPI `create_trained_policy`，配置
与官方 `pi05_libero` 发布一致：`pi05=True`、horizon 10、不使用离散状态输入、
关闭额外 delta transform；该配置只用于原生测试，不修改 RoboDojo 的默认 deploy。

验收工具默认连续推理两次；检查动作非空、二维、每个数值有限，G05 输出宽度与
自身 processor 一致，π0.5 输出为 10×7。输入是合成 RGB/状态，不声称任务成功。
每次保存动作 `.npz`、版本、设备、source commit、测试脚本 SHA-256、加载/推理耗时
和 JSON 报告，失败时另存 traceback。`--output` 必须是新目录，以免覆盖已有证据。

### π0.5 原生推理实际结果

2026-11-24 21:01（Asia/Shanghai）完成，进程退出码 0：

| 项目 | 结果 |
| --- | --- |
| checkpoint | 官方 `gs://openpi-assets/checkpoints/pi05_libero` |
| 下载完整性 | 16/16 文件通过长度及 MD5/CRC32C 校验；约 12.44 GB |
| GPU | 物理 GPU 1，RTX 4090；进程内为 `CudaDevice(id=0)` |
| 权重/模型加载 | 85.08 秒 |
| 第 1 次推理 | 66.19 秒，包含首次 JAX 编译/算子选择；输出 10×7，全部有限 |
| 第 2 次推理 | 0.102 秒，输出 10×7，全部有限 |
| 总耗时 | 151.37 秒 |
| 任务成功率 | 未测量；输入为合成 RGB/状态，未启动 LIBERO 或 RoboMME |

原始报告：[stage1b-pi05-native.json](validation/stage1b-pi05-native.json)。
可复现的源对象 generation 和校验值：[stage1b-pi05-assets.json](validation/stage1b-pi05-assets.json)。
本地动作数组在 `runs/native-pi05-01/actions_0.npz` 和 `actions_1.npz`。
报告记录实际启动时的 source commit 与入口脚本 SHA-256；测试是在开发分支工作区执行，
以脚本 SHA-256 标识该次入口版本。本机当时有其他权重下载，这些耗时仅作为运行记录。

### G05 原生推理实际结果

2026-11-24 21:20（Asia/Shanghai）完成，进程退出码 0：

| 项目 | 结果 |
| --- | --- |
| checkpoint | `OpenGalaxea/g05-robodojo` 官方 FM-only 发布包 |
| checkpoint 完整性 | archive 34,425,477,120 bytes，官方 SHA-256 校验通过 |
| processor 完整性 | 官方训练资产 archive 13,699,174,400 bytes，SHA-256 校验通过；只提取匹配 HF processor |
| GPU | 物理 GPU 0，RTX 4090 |
| 权重/模型加载 | 404.54 秒，包含本机机械盘上的冷启动依赖读取与初始化 |
| 第 1 次推理 | 5.593 秒 |
| 第 2 次推理 | 0.792 秒 |
| 动作输出 | 左右臂各 32×6，左右夹爪各 32×1，两次输出全部有限 |
| 总耗时 | 410.94 秒 |
| 模式 | FM、3 cameras、obs_steps=1、action_steps=32；上游 SDPA fallback，关闭 torch.compile |
| 任务成功率 | 未测量；使用合成输入，未启动 RoboDojo 或 RoboMME |

原始报告：[stage1b-g05-native.json](validation/stage1b-g05-native.json)。
权重与 processor 的来源和校验值分别见
[stage1b-g05-assets.json](validation/stage1b-g05-assets.json)、
[stage1b-g05-processor-assets.json](validation/stage1b-g05-processor-assets.json)。
本地动作数组在 `runs/native-g05-01/actions_0.npz` 和 `actions_1.npz`。

G05 checkpoint 的上游配置会提前解析仅供训练使用的数据集环境变量。
原生验证入口在该变量未设置时给出空值，实际没有打开或使用训练数据；
模型、normalizer、tokenizer 和输入 schema 均来自官方 checkpoint 与配套 processor。
配置中的缺失 `tokenizer_meta` 使用上游 YAML key_dims fallback，未自行修改解码器。

阶段 1 最终 [33 项回归测试输出](validation/stage1-final-regression.txt) 全部通过，
包括 27 项原有回归和 6 项资源下载/中断恢复回归。
阶段 1A 的 32 项记录保留为当时的历史证据。

π0.5 原生测试使用的 LIBERO checkpoint 输出 7 维末端增量动作；现有 Pi_05 adapter
默认配置为 RoboDojo 双臂 14 维关节动作，不能直接裁切、补零后当作接口通过。
下一步先验证已有匹配 checkpoint 的 G05 adapter，再为 RoboMME 使用其官方
π0.5 baseline 权重与 Panda 单臂接口单独验证。
