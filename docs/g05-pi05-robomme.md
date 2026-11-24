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
| 1 | 原生模型测试 | 两套隔离依赖；官方权重校验；直接调用上游推理，输出非空、有限、形状正确的动作；保存机器可读记录 | 进行中 |
| 2 | XPolicyLab 适配验证 | 真实模型加载；无模拟器 debug 闭环；普通 RGB 与共享编码 RGB 两条路径通过 | 待阶段 1 通过 |
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
