# KnowAct — XPolicyLab 实验版

> **版本：** V7.1 experiment-ready cleanup  
> **定位：** 基于 G0.5 的原生多模态上下文与闭环动作策略研究实现。  
> **开发原则：** 主干不变、原生上下文、共享 LM Head、统一 CE；没有实验证据时不增加新的模型实体。

本目录可直接放入 XPolicyLab 根目录：

```text
XPolicyLab/
└── policy/
    ├── G05/                 # 原始 XPolicyLab G05 Baseline，不修改
    └── KnowAct/             # 本项目
        ├── G05/              # KnowAct 使用的 vendored G0.5 源码副本
        ├── model.py          # XPolicyLab Policy Adapter
        ├── deploy.py         # 持续上下文与执行回执
        ├── train.sh          # 训练入口
        ├── eval.sh           # 仿真评测入口
        └── compare.py        # 成对评测计划与统计
```

`KnowAct/G05/` 是 KnowAct 使用的 G0.5 源码副本，不是原始 Baseline。**不要将其复制覆盖到 `policy/G05/`。** 原始 G05 和 KnowAct 应使用独立 Python 环境、独立 policy-server 进程，因为两者都导入名为 `g05` 的 Python 包。

## RoboMME / Panda

已接入四任务 Panda 关节控制训练，支持官方 Qwen3.5-0.8B 和 G05 两种初始化。环境客户端位于仓库的 `benchmarks/robomme/`。数据准备、两阶段权重、四卡启动命令、原生评估、检查点结构与实测记录见 [RoboMME 训练手册](../../docs/experiments/knowact-robomme/README.md)。

## V7.1 原始发布状态

| 项目 | 当前状态 |
|---|---|
| G0.5 原生模型、视觉/状态编码、ActionCodec | 复用 |
| 多模态上下文追加与完整/增量计算 | 源码级 CPU 回归通过 |
| 文本 AR 与动作 BAR16 | 已接入同一个 `InferenceState` |
| 多轮训练 | 已接入同一 Episode 历史采样；当前监督目标为 CE |
| 执行回执与已执行动作历史 | 已接入；按 chunk 边界重规划 |
| XPolicyLab Adapter / RoboDojo 入口 | 已接入 |
| 正式预训练权重训练与 GPU 验收 | **尚未完成** |
| 跨 Episode 示范式 ICL 数据管线 | **尚未完成** |
| 多轮 Validation DataLoader | **尚未完成；当前验证仍走原单轮路径** |
| 真实 RoboDojo 成对成绩 | **尚未产生** |

当前发布包包含 V7 的真实源码改动和实验记录，以及本次整理新增的文档、回归测试和比较脚本修正。历史实验中使用过小尺寸模型、随机初始化权重、合成任务和模拟 IO；这些结果不能解释为正式预训练 G0.5 的性能提升。完整边界见 [`docs/08_当前证据与限制.md`](docs/08_当前证据与限制.md)。

## 从哪里开始

1. [项目范围、版本基线与原则](docs/00_项目说明.md)
2. [V7.1 代码整理发布说明](docs/RELEASE_SUMMARY.md)
3. [V7 原始开发与实验报告](EXPERIMENT_REPORT.md)
4. [代码地图与模型架构](docs/01_代码地图与架构.md)
5. [数据语义与多轮训练](docs/02_数据与训练.md)
6. [推理状态、BAR16 与闭环回执](docs/03_推理与闭环.md)
7. [正式实验矩阵与消融设计](docs/04_实验计划.md)
8. [XPolicyLab 成对 Benchmark 协议](docs/05_对比评测.md)
9. [安装、训练、测试和评测复现](docs/06_复现手册.md)
10. [验收门槛与统计规范](docs/07_验收与统计规范.md)
11. [当前证据、已知限制和验收门槛](docs/08_当前证据与限制.md)
12. [原始结果文件索引](results/README.md)
13. [实验记录模板](docs/10_实验记录模板.md)

## 快速检查

在已安装 XPolicyLab 和 vendored G0.5 依赖的独立环境中，从 XPolicyLab 根目录执行：

```bash
export XPL_ROOT="$PWD"
export KNOWACT_ROOT="$XPL_ROOT/policy/KnowAct/G05"
export KNOWACT_PYTHON="/path/to/knowact-env/bin/python"

cd "$XPL_ROOT/policy/KnowAct"
bash -n train.sh eval.sh install.sh
python -m py_compile model.py deploy.py compare.py preflight.py
python preflight.py --output results/preflight.json
```

完整 73 项 CPU 源码回归命令、正式资产准备和训练/评测命令见 [`docs/06_复现手册.md`](docs/06_复现手册.md)。`preflight.py` 仅检查路径、Python/CUDA 和目录结构，不会加载 checkpoint 或验证数据内容。

## 关键默认值

| 参数 | 当前值 | 说明 |
|---|---:|---|
| Action generation | AR | Native Context 当前只支持 AR |
| BAR block size | 16 | 新训练布局；不能对已有 checkpoint 只改推理参数 |
| Action horizon | 32 | RoboDojo task 配置 |
| Replan / dispatch stride | 16 | 生成 32 步，默认只下发前 16 步 |
| Native context turns | 4 | 每个样本随机取 1 到 4 个同 Episode 历史轮次 |
| Context token budget | 16,384 | 硬上限；超限报错，不静默裁剪 |
| Loss | 原生 CE | 历史轮次作为上下文，当前轮次监督；不新增记忆 Loss |

## 许可与来源

XPolicyLab 快照标签记录为 `4a9e3afd73b86a48c5db076f5cf9dbfbef1a180e`。用户提供的 ZIP 不包含 `.git` 元数据，因此此处是集成记录中的快照标识，不是本包重新执行 `git rev-parse` 得到的结果。`G05/` 内保留上游许可证、NOTICE 和第三方声明；使用和再分发前请遵守对应许可。
