# KnowAct V6 — 源码接入与验收报告

## 1. 范围与基线

开发仅在 `policy/KnowAct`，原版 `policy/G05` 保持不变。基于上传的 XPolicyLab 快照及 V5。无新增可训练模型参数。没有官方 checkpoint、实际训练数据集、GPU 或真实 RoboDojo 环境。CPU 实验使用缩小的实际 G05ModelQwen35（Vision、Proprio、GDN、Full Attention、MRoPE）、G05Policy、InputPreprocessor、BARBuilder 和真实小型 ActionCodec（FSQ 配置，随机初始化；不是发布版预训练 RVQ 权重）。部分缺失依赖用测试构造器/导入替身；Policy 测试通过 `__new__` 配置对象，不验证完整 checkpoint 初始化。文本使用测试映射，XPolicyLab 测试使用真实 adapter/inferencer 和模拟处理器、RPC 与环境。没有以模拟结果替代机器人成功率。

## 2. 精简实现

- `G05/src/g05/models/g05/native_context.py`：复用 V5 `encode_context_segment` / `forward_context`、原 InferenceState、SparseKVCache、原 CE。只组织历史追加、语言 AR、动作 BAR 与完整 Episode 监督；无独立 Memory/Mode Head。
- `g05_policy.py`：原 `forward` / `prefill` / `generate_text` / `generate_action` 路由至 native 路径。仅在完整 ActionCodec 解码、shape 与有限数值检查通过后提交候选。生成失败、超预算、重复动作请求不污染已提交状态。
- BAR16 固定布局：直接由原训练模板构造 BOS、分组码、PAD、EOS 和文本后缀，沿用位置约束与联合规则；修正分组标记后的夹爪偏移。不是任意长度生成，不重写 Codec，不使用未验证的时间因果解码。固定格式约束不保证块内联合运动分布正确。
- `native_episodes.py` / `base_lerobot_dataset.py` / `mixture_lerobot_dataset.py` / `scripts/finetune.py`：原数据集、collate、优化器和训练入口。随机 1–4 轮同 Episode 历史，按完整 action horizon 非重叠索引、拒绝未来观察与不合规偏移、跳过 padded 历史、按监督 Token 加权 CE。尚未用实际 LeRobot 数据和完整分布式 Trainer 验证；独立 Episode 前向不是 GPU packing。
- `inferencer.py`、外层 `model.py`、`deploy.py`：每个环境独立 InferenceState，当前观测+历史预测计划+实际已下发并完成的控制步数（不是测得的完整执行轨迹），拒绝过期/重复回执与超出派发步数；批次原子提交、单/多环境 reset。只支持同步 chunk 边界重规划和缩短派发区间，不支持物理控制中的异步打断。新观测含真实环境返回的当前状态。
- `compare.py`：原 XPolicyLab 入口成对运行；同一 KnowAct checkpoint 的历史清空消融；dry-run 清单和配对 CSV 统计。没有真实评测结果。

## 3. 实际工程测试

共 54/54 通过：V5 回归 27 项 + 新增 Policy/Codec/episode 19 项 + XPolicyLab 交互 5 项 + 对比脚本 3 项。见 `results/final_regression.log`。真实小型模型 BAR16：39 个有效动作码、48 个填充后码位，解码 shape `[1, 8, 5]`。测试覆盖完整与增量前向、MRoPE、两轮布局/缓存重放、原 CE 反向传播、视觉及状态梯度、非法解码回滚、停止边界、预算、历史指令、双夹爪语法偏移、不同环境隔离、batch 失败原子性、receipt 及 reset。文本生成 EOS 测试强制 logits 验证协议，不是训练过的语言能力。BF16/GPU 仍未正式验收。

**关键负面结果：** 修改小型原生 Codec 后部合法神经码，前半段已解码轨迹最大绝对变化 0.0016345679759979248（归一化动作单位）。一次反例足以否定该测试配置的严格时间前缀稳定性。不能外推为所有正式码本的误差大小。部署先完整解码，再执行并重新观察，不在未完成 RVQ 组上直接执行。

## 4. 实际训练对照（未达能力验收）

相同随机小模型、相同监督目标数、每个种子 100 次 AdamW 更新；两个反复训练的合成情景，当前指令/观测相同，仅过去指令提供区别；历史组额外带已知示范，非模型自行 rollout。输入 Token 与 FLOPs 不等。每种配置三个种子、每种子两个训练分布内情景；不属于 held-out ICL 或泛化测试。

| Seed | 仅当前 CE | 历史 CE | 仅当前码准确率 | 历史码准确率 | 仅当前夹爪规则 | 历史夹爪规则 |
|---|---:|---:|---:|---:|---:|---:|
|11|0.172910|0.128801|69.23%|70.51%|0%|0%|
|22|0.182368|0.167165|89.74%|89.74%|0%|50%|
|33|0.135532|0.117135|79.49%|82.05%|0%|0%|
|平均|0.163604|0.137700|79.49%|80.77%|0%|16.67%|

两组自由生成完整动作码精确率均为 0%。低 CE 不足以证明正确动作生成；历史组也没有学到可靠的条件控制。不能据此宣称超越 G05。保留逐步 loss、预测码和连续夹爪输出于 `results/training_results.json`。下一步需诊断实际监督位置、量化规则、BAR 联合依赖、teacher-forcing/free-run 差异，再用实际数据和预训练权重测试。

## 5. 正式比较协议与剩余风险

同任务、同测试实例与随机种子、同归一化/动作空间、同执行间隔、匹配训练预算。G05 官方 AR（checkpoint 本身决定 BAR/AR 与观测布局），KnowAct BAR16 native，KnowAct_reset 同权重无历史；后续增加历史训练长度消融、同 BAR16 配置的 current-only 训练对照，避免将宽度变化误归因于 memory。当前 launcher 匹配原 baseline 的 16 步执行间隔；原 native 训练使用完整 nonoverlap horizon（通常 32），短间隔重规划属于额外分布偏移，不能声称训练推理完全对齐。正式对照需报告原 baseline 的合适历史配置、真实训练 token 与 GPU 时间、成功率、延迟/吞吐/显存、干扰与恢复任务、跨本体验证。脚本只生成清单与运行日志，实际 episode outcomes 需按原 benchmark 导出并配对。

未验收：正式 HF processor 与发布版 ActionCodec/预训练参数加载；实际 LeRobot 视频/样本/分布式训练；GPU BF16 / FlashAttention / FLA；复杂长程真实反馈与 ICL；原 Codec 因果前缀解码；真正异步连续控制；超预算上下文管理；完整 RVQ 联合分布质量。当前没有引入替代 Codec 或额外网络来掩盖这些问题。

## 6. 复现与文件

安装、训练与运行命令见 `README.md`。`results/final_regression.log` 为最终测试；`training.log`/`training_results.json` 为六次实训输出；`benchmark_manifest.example.json` 与 `benchmark_dry_run.log` 为演示清单（路径占位符，无真实模型评分）。保留原许可证及 V5 源码回归报告。
