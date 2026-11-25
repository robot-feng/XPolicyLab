# KnowAct V7.1｜代码整理与文档发布说明

## 1. 发布范围

本发布是 XPolicyLab 中 `policy/KnowAct/` 的可复现实验整理版，不包含整个 XPolicyLab 仓库，也不包含官方模型权重或 RoboDojo 数据集。解压时将 `policy/KnowAct/` 覆盖/合并到你自己的 XPolicyLab 根目录即可；不要覆盖 `policy/G05/`。

## 2. 代码整理

- `compare.py`：拆分为模型变体构造、命令计划、paired CSV 校验、cluster bootstrap 和 CLI；默认官方 RoboDojo G05 action source 为 FM；AR 需要显式指定；G05-MEM 可独立指定 source/offsets；默认要求 G05 与 KnowAct 的实际执行步长匹配。
- `preflight.py`：明确区分路径/环境存在性检查和真正的模型加载验收；检查 Python 版本、CUDA、XPL/G05 源码、checkpoint、processor、ActionCodec 和 LeRobot 目录。
- `model.py`：格式化 Adapter 代码，明确每环境状态、G05 模块路径隔离、BAR16 checkpoint 检查、stride 对齐和批量原子提交。
- `deploy.py`：格式化单环境/批量执行回执流程，明确只在 chunk 边界重新规划。
- `native_context.py`：格式化原生 Context 编码、append、文本 AR、BAR16、执行前缀提交和 CE 训练路径；未增加网络或参数。
- `native_episodes.py`：格式化 same-Episode 历史采样、边界检查、hold-pad 与 Collate。
- `audit_action_codes.py`：明确输入必须是同一上下文下的 paired expert alternatives，避免把不配对轨迹当作多峰联合分布证据。
- `train.sh`：实验名与训练模式解耦；默认 AR-only，非法训练模式明确报错。
- 移除 Python bytecode / `__pycache__`，保留原始实验数据、日志和历史失败记录。

## 3. 文档

`docs/` 现在覆盖项目范围、代码地图、数据与训练语义、推理与闭环、实验矩阵、对比评测、复现、统计验收、证据边界、结果索引和实验记录模板。历史 V5/V6 方案移入 `docs/history/`，避免旧补丁被误应用到当前源码。

## 4. 比较协议的关键修正

XPolicyLab 官方 RoboDojo G05 simulation checkpoint 是 FM-only。因此 `compare.py` 的 G05 默认 action source 改为 `fm`；如果使用真实 G05-AR checkpoint，必须显式传 `--g05-action-source ar`。这避免把不匹配的 FM checkpoint 强制当成 AR baseline。

## 5. 验证结果

本次整理后的 CPU 源码回归：**73/73 passed**。另外通过：

- Python syntax / compile 检查。
- `train.sh`、`eval.sh`、`install.sh` shell syntax 检查。
- compare dry-run manifest 检查（G05=FM、KnowAct=AR、执行步长匹配）。
- paired CSV 的重复 episode、缺失配对和非法 success 值拒绝测试。
- Markdown 本地链接检查。
- ZIP 完整性检查。

这些是 CPU 源码/模拟 IO 测试，不是正式权重、GPU 或 RoboDojo 仿真成绩。

## 6. 尚未完成

- 官方预训练 checkpoint / HF processor / ActionCodec 的真实加载回归。
- GPU BF16 / FLA / FlashAttention 回归。
- Native multi-turn validation DataLoader。
- 跨 Episode demonstration ICL 数据管线。
- 正式 RoboDojo paired benchmark。
- 真实预训练 ActionCodec 的依赖与因果前缀审计。

这些限制在 `docs/08_当前证据与限制.md` 中逐项列明。正式性能结论必须等真实权重、真实数据和同预算仿真对照完成后再给出。
