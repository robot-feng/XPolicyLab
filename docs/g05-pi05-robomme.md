# RoboMME 环境接入

本分支从本地初版 `4a9e3afd` 建立。RoboMME 客户端负责创建 Panda 模拟器、读取观测、执行动作和记录原生判定。策略服务独立启动，加载 checkpoint 并推理。连接地址可以是本机或另一台机器。

```text
环境机：RoboMME → TASK_ENV ↔ 所选策略的 deploy.py → XPolicyLab WebSocket
策略机：XPolicyLab WebSocket → 所选策略的 model.py → checkpoint
```

初版的 `deploy.py` 是环境端可调用的部署循环，`model.py` 是服务端的模型实现。新客户端把 Panda 封装为 `TASK_ENV`，按该策略的 `deploy.yml` 选择单环境或 batch 部署循环。batch 模式每次有一个活动环境，episode 按顺序运行。适配范围为 WebSocket、Panda 单臂、绝对关节控制以及使用标准观测和动作 RPC 的部署循环。

## 在环境机运行

1. 按 [RoboMME 安装指南](https://github.com/RoboMME/robomme_benchmark) 安装模拟器、任务数据和资产。本机使用同级目录 `../robomme_benchmark/.venv`，其中已有 ManiSkill、SAPIEN、imageio 和 FFmpeg。该 Python 还需要本仓库 `pyproject.toml` 中的通信和数据依赖。
2. 在父工作区注册 Panda。给 `../env_cfg/robot/_robot_info.json` 添加键 `"panda": {"arm_dim": [7], "ee_dim": [1]}`，保留其他机器人。仓库中的 `utils/robot/_robot_info.json` 已包含同一项。`../env_cfg/panda.yml` 的内容可用 `config: {robot: panda, sim: robomme}`，对应 `../env_cfg/sim/robomme.yml` 为 `scene: {num_envs: 1}`。
3. 在策略机用该模型自己的入口启动兼容的 Panda 策略服务。环境机使用其地址和该策略的部署循环。从 XPolicyLab 根目录运行下面的命令。示例假设一个采用 `Pi_05/deploy.py` 的兼容服务正在本机端口 6000 监听。

```bash
EVAL_ENV_TYPE=sim ROBOMME_EPISODES=1 ROBOMME_MAX_STEPS=1300 \
bash policy/Pi_05/setup_eval_env_client.sh \
  RoboMME PickXtimes external panda joint 0 0 \
  "$(cd ../robomme_benchmark/.venv && pwd)" \
  ckpt_name=external,action_type=joint 6000 localhost
```

`external` 是原有启动接口的 checkpoint 标签。权重和模型 seed 由策略服务的启动命令决定。只测试接线时，也可以用仓库的 `demo_policy` 服务及其同名客户端，把 `ROBOMME_MAX_STEPS` 设为 2。原版 G05 RoboDojo checkpoint 输出双臂 ARX 动作，原版 Pi_05 默认配置也是双臂。运行上述单臂任务需在服务端选择具备对应输入和动作空间的模型。[此前已验证的官方 π0.5 权重、服务安装与启动记录](https://github.com/robot-feng/XPolicyLab/blob/59fe99687c60568c78eb6f54c77b8062740664e9/docs/g05-pi05-robomme.md)

| 环境参数 | 默认值 | 用途 |
| --- | --- | --- |
| `ROBOMME_SPLIT` | `test` | 可用 train、val、test、ood_test |
| `ROBOMME_START_EPISODE` / `ROBOMME_EPISODES` | 0 / 1 | 当前任务的首个 episode 和连续回合数 |
| `ROBOMME_MAX_STEPS` | 1300 | 传给原生 builder 的预算，报告以实际终止步数为准 |
| `ROBOMME_ACTION_STEPS` | 16 | 每个标准推理 chunk 最多执行的动作数 |
| `ROBOMME_EVALUATION_ID` | 结果目录名 | 跨客户端复用同一服务时，在第一次连接和后续各次指定同一 ID |
| `ROBOMME_OUTPUT_DIR` | `runs/robomme/<id>/` | 新建结果目录，保存 report.json 和每回合的 MP4 |

每次指定一个原生任务名。对多个任务依次启动客户端时，设置相同的 `ROBOMME_EVALUATION_ID`。服务在首次握手时绑定该 ID，各任务的 trial ID 包含任务、split 和回合编号。直接调用的等价入口可以用 `python -m XPolicyLab.utils.robomme_env_client --help` 查看。需要从 checkout 或其父目录导入 XPolicyLab。

环境端将前视和腕部的 RGB 映射到 `cam_head`、`cam_wrist`。状态是 7 个关节和第一根夹指的物理位置，动作是 7 个绝对关节目标和一个夹爪命令（−1 闭合，+1 张开）。策略 reset 后，先发送演示帧，再将当前帧交给部署循环。环境使用数据集的 episode seed，并按原生 `terminated`、`truncated` 和 `info["status"]` 记录结果。

## 对上一轮改动的复核

范围为初版 `4a9e3afd` 到实验提交 `59fe9968`。下表覆盖该比较中的 55 个文件。完整逐文件 diff 和各阶段的原始记录保存在[实验分支的固定版本](https://github.com/robot-feng/XPolicyLab/tree/59fe99687c60568c78eb6f54c77b8062740664e9)。

| 原改动 | 文件数 | 复核与处理 |
| --- | ---: | --- |
| `policy/Pi_05/` 下的 model、robomme、install_robomme、deploy.yml、README | 5 | `robomme.py` 实际做的是官方模型加载。`model.py` 根据 benchmark 切换 OpenPI 源码、编码器和 reset 行为，把环境选择绑定到权重实现。整个模型实验留在旧分支。 |
| `policy/G05/` 下的 model、server launcher、install、requirements、deploy.yml、README | 6 | 推理依赖和 checkpoint 解析对原版 G05 验证有用。双臂保护、状态打包、历史解码和训练说明属于该策略自身，环境接入可独立交付。 |
| `scripts/` 的下载器、原生推理脚本、metadata 安装器 | 3 | 服务首次准备和原生验证使用了这些脚本。环境端使用已有服务，Panda 注册步骤如上。 |
| `utils/robomme_env_client.py`、`run_robomme_env_client.sh`、`resolve_eval_env_type.sh`、`setup_env_client.sh`、`robot/_robot_info.json` | 5 | 保留环境转换与 Panda 注册。改用既有 sim 模式和共享分派入口，移除新增模式和单独 shell 转发。旧客户端自行执行全套推理循环，现调用策略的原有部署循环。 |
| `utils/process_data.py`、`activate_eval_env.sh`、`run_debug_env_client.sh` | 3 | 递归解码用于 G05 旧历史路径；RoboMME 发送原生 RGB。统一激活及 debug 变更会影响其他 benchmark，现只在 RoboMME 入口选取解释器。 |
| tests 下的 robomme、eval_metadata、model_assets、nested_observations | 4 | 保留环境合同和已有 deploy 循环的测试。模型、下载器和通用解码的测试随各自实验保留在旧版本。 |
| 两份根 README、阶段指南、25 个 validation 文件、.gitignore | 29 | 中英文 README 各增加一个环境入口。用本页记录用法、范围和验证。下载目录规则、历史日志、模型数值对照仍可按旧提交复查。 |

初版对比命令（在包含两个提交的仓库执行）：

```bash
git diff --stat 4a9e3afd 59fe9968
git diff --stat 4a9e3afd HEAD
git diff --exit-code 4a9e3afd HEAD -- policy/ client_server/ setup_policy_server.py utils/process_data.py
```

## 本分支的实际验证

2026-11-25，从各策略原有的 `setup_eval_env_client.sh` 进入，新分支上的客户端和独立策略进程通过 WebSocket 通信。π0.5 服务从实验提交 `59fe9968` 的独立 checkout 启动，使用已验证的官方 Panda 权重；客户端与两个策略的部署循环均来自本分支。

| 服务与设置 | 环境结果 | 本机记录 |
| --- | --- | --- |
| 初版 demo_policy，单环境 RPC，PickXtimes/test/0，预算 2 | 3 步，原生 timeout | `runs/robomme/env review demo/report.json` |
| π0.5 官方 baseline，batch RPC，PickXtimes/test/0，预算 1300 | 422 步，原生 fail | `runs/robomme/env-only-verified-PickXtimes/report.json` |
| 同一 π0.5 服务，VideoRepick/test/0，预算 1300 | reset 返回 258 帧，108 步，原生 fail | `runs/robomme/env-only-verified-VideoRepick/report.json` |

π0.5 的两个回合成功数为 0/2。与实验版本在相同权重、模型 seed 0 和执行长度 16 下的记录逐个比较，原生终止状态、步数及两个 MP4 文件的 SHA256 都一致。首次跨任务重连时，新的评估 ID 被已有服务拒绝。表中复测使用同一 evaluation ID 及各回合独立的 trial ID。

`python -m unittest discover -s tests` 共通过 29 项（初版 27 项、本次新增 2 项）。新增测试运行原有单环境和 batch 部署循环，覆盖演示帧顺序、RGB、夹指单位、动作 chunk 截取和终止。另用相同输入对比初版和当前共享分派脚本，RoboDojo sim、RoboMME debug、RoboDojo real 三种情况下目标脚本和传入参数一致。
