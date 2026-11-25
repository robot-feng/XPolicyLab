# KnowAct V5 — 原生 G0.5 源码级接入（研究原型）

基线是用户提供的 `XPolicyLab-main (1).zip` 内 `XPolicyLab-main/policy/G05/G05/`，而不是独立的 V1–V4 Toy 主干。未修改原 ZIP。保留原 LICENSE-G0.5、NOTICE 和第三方许可；本修改用于非商业研究，使用者仍需遵守原许可，不代表 Galaxea 官方发布或认可。归档的 SHA-256 和改动文件见 `tests/results/environment.json`。

## 已完成范围

- 实际 G0.5 的 Qwen3.5 模型、ViT、状态 MLP、GDN、Full Attention、MRoPE、BARBuilder 和 CE；不添加可训练记忆网络、Head、Loss 或独立 Session 类。
- GDN 多 Token 卷积接续，按层读取历史状态；读取缓存但不提交时不修改卷积历史。
- 普通增量注意力因果约束；已知 BAR 输入块仍可双向访问。完整和后缀使用统一因果组规则。
- `SparseKVCache.fork()`：KV append 由原方法创建新的张量，分支共享旧 KV；独立复制循环/卷积状态。`detach()` 保留卷积状态。
- `BuiltinActionProcessor` 按 `is_masked` 屏蔽历史动作监督；不改变历史输入或 BAR 标签排列。
- `BARBuilder` 普通 ID / 字符串路径防止固定块数导致静默截断，以及 fractional group ID 溢出。未修改专用 serializer 的所有分支。
- `G05ModelQwen35.encode_context_segment()` 与 `forward_context()`：直接调用原生编码器和模型，支持独立编码视觉事件、状态事件、文本或动作；原生完整 MRoPE 元数据重算、后缀注意力计算、缓存分支保护。新路径只支持无 padding 的 episode、同一 batch 共享图像 grid 布局、确定性 pi0fast/lyc 位置规则；不能切开一个图像/BAR 输入组。
- `configs/tokenizer/actioncodec.yaml` 新训练默认 BAR 宽度 16。原有 task / checkpoint 配置没有改动。任务覆盖可能仍关闭 BAR，必须显式开启；不能直接将已训练 checkpoint 的生成宽度改为 16。

## 用法：原生计算接口

已有的 `InputPreprocessor.preprocess(samples=..., control_flag=None, ...)` 可以逐个 turn 构建 IDs / labels / typed masks；上下文示范和真实执行记录应明确设置 `-100` 监督屏蔽，不能把还没执行的未来 GT 动作写入历史。每个 turn 的图像应按其占位顺序放入 camera 字典；在原生上下文模式下禁用旧视频时间融合，逐次独立编码。以下为接口示例，实际需先加载官方模型、tokenizer、归一化配置和准备数据。不要修改缓存对应的前缀内容或模型权重后继续 append。

```python
# 一次模型初始化后，在既有训练/推理流程中使用。
# model 是 G05ModelQwen35。
parts = []
masks = []
grids = []
labels_all = []
for turn in turns:
    # 使用原来的 InputPreprocessor 产生输入，不新增 token 词表。
    ids, labels, mask = processor.preprocess(
        samples=turn['samples'], control_flag=None, device=device,
        training=model.training,
    )
    emb, grid = model.encode_context_segment(
        ids, mask, turn.get('pixel_values'), turn.get('proprio'),
    )
    parts.append(emb)
    masks.append(mask)
    grids.extend(grid)
    labels_all.append(labels)
# 整条 Episode 共享一次 VLM forward + 原生 CE；不是新的 loss。
hidden, _, _ = model.forward_context(torch.cat(parts, 1), torch.cat(masks, 1), grids)
loss, accuracy = model.ar_helper.cal_ce_loss(hidden, torch.cat(labels_all, 1), model)
```

增量计算用同一函数，新增 suffix embeddings、完整 typed mask、完整 grid 元数据和前一个 cache；只有结果确认后才替换旧 cache。当前仅在模型计算层验证；没有把旧 `G05Policy.generate_action()`、服务器与 episode dataloader 接到新接口，不能当作正式全双工控制器。需要实际 `ActionCodec` 权重时，必须按官方路径加载。新的 cache 记录布局，拒绝已变化的 mask/grid 前缀或 legacy cache；不验证历史 raw 内容，调用者必须遵守 append-only 语义。需要删除/编辑历史时从保留序列重新 prefill。

## 真实源文件修改前后对照

使用同一随机初始化参数、同一输入、实际 PyTorch GDN 和 mask/processor，分别导入原始快照和修改后文件。测试 27 Token，分块 7+16+4。

| 项目 | 原始 | 修改后 |
|---|---:|---:|
| GDN hidden 最大绝对误差 | 0.5658478141 | 5.9604645e-8 |
| GDN recurrent 最大绝对误差 | 3.8586557e-5 | 9.3132257e-10 |
| 最终卷积状态误差 | 0 | 0 |
| 4 个普通追加 Token 内错误开放的未来边 | 6 | 0 |
| 18 个动作代码的 masked BAR16 输入中有监督标签数 | 51 | 0 |

最终卷积状态相同不说明中间卷积正确；最后一块长度达到卷积核长度时会覆盖最终历史，错误发生在块开始处。

## 回归测试

`tests/test_native_context.py`: 27/27。GDN 测试包括 B=1/2、卷积核 1/2/4/7、4 种分块，另有真实 ViT、混合分辨率多相机、状态编码、BAR 1/8/16/32、未来标签隔离、前缀修改拒绝、真实 CE 梯度、编码器反传、旧单轮 prefill 与新接口对照、分支/会话隔离。

三种子 68 Token 事件序列：FP32 hidden max = 5.126e-6，logits max = 4.184e-5，cache max = 7.629e-6；MRoPE 和 top1 完全一致。实际代码使用的是缩小版 4 层（3 GDN + 1 Full Attention），不是官方 24 层权重。CPU BF16 autocast 诊断中，最大 logits 误差为 0.625，hidden max 0.08294，top1 在这些特定样本一致，但不能满足严谨数值一致性门槛，尚未验收。测试记录 `precision.json`。尝试强制 fallback matmul 关闭 autocast 没有改善端到端误差，未保留该修改。

## 实际源码的小模型训练 smoke test

`train_source_smoke.py` 直接训练上面缩小版真实模型（VLM + vision + proprio 参数 77,380，未使用 FM expert）；实际原生 ViT、状态 MLP、BARBuilder、CE。任务为从过去的四类指令恢复当前 16 个动作代码，每类的代码具有位置相关结构；当前图像和状态独立于类别，只有历史指令提供信息。使用合成动作 ID，不运行真实 ActionCodec，不是物理轨迹或未见任务 ICL。B=8，三随机种子，AdamW lr=0.003，360 steps；另有第一次 120 步的失败记录。训练后的测试不提供未来 GT action block，实际自由预测 BOS block 的 16 个 logits，然后将生成的块提交预测 EOS。只有固定结构化规则与固定类别的训练/测试，不是新规则泛化。

| 训练步 | Seed | 单 Token 准确率 | 动作类别 | 完整块准确率 | EOS 整块准确率 | 反事实配对均完全正确 |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 11 | 73.4375% | 100% | 0% | 100% | 0% |
| 120 | 22 | 76.5625% | 100% | 0% | 100% | 0% |
| 120 | 33 | 75.0% | 100% | 0% | 50% | 0% |
| 360 | 11 | 96.875% | 100% | 50% | 75% | 0% |
| 360 | 22 | 89.0625% | 100% | 0% | 50% | 0% |
| 360 | 33 | 100% | 100% | 100% | 100% | 100% |
| 360 | 平均 | 95.3125% | 100% | 50% | 75% | 33.3333% |

“动作类别”指代码值对应预定义四类的逐 Token 分类准确率；不等于完整动作预测。移除历史的整块准确率为 0%，这只是缺失必要信息的诊断，不是公平的无历史训练基线。当前 360 步训练结果显示完整块精度和种子稳定性不足，即使 CE 降低且动作类别全正确，仍不能可靠生成结构化动作。不能据此宣称优于真实 G0.5。保存完整 JSON 和日志，未保存本次小模型权重（可通过脚本重训）。

## 复现

从此 G05 根目录运行：

```bash
# CPU standalone（本次实测）：需 torch, omegaconf, einops, numpy 等
PYTHONPATH=tests python tests/test_native_context.py
PYTHONPATH=tests python tests/repro_original.py
PYTHONPATH=tests python tests/train_source_smoke.py
# 原始基线：指向原始未改动的 G05 子项目根目录
G05_TEST_SOURCE=/path/to/original/G05 PYTHONPATH=tests python tests/repro_original.py
# 在官方依赖环境尝试普通包导入（尚未实测完整官方环境）
G05_TEST_STANDALONE=0 PYTHONPATH=tests python tests/test_native_context.py
```

运行环境 Python 3.13.5 / torch 2.10.0 CPU；官方 pyproject 要求 Python 3.10.x / torch 2.7.1；缺少 Transformers、FLA、FlashAttention、Liger 和真实 checkpoints。CPU loader 只绕开 policy 包的 eager imports、关闭训练监控并在 Transformers 不存在时提供空 attention registry，实际 eager attention、GDN、Vision、state MLP、CE 全部来自上传源码，没有用替身数学模型。真实 tokenizer/codec checkpoint 不在本次 ZIP 内。

## 正式验收下一步（未完成）

1. 在官方 Python/CUDA 环境安装所需依赖，加载 `checkpoints/g05-base/checkpoints/model_state_dict.pt`、`checkpoints/action_tokenizer.pt`、`checkpoints/qwen3_5_2b_base_processor/`，保持原 checkpoint config 和数据归一化文件，完成原版 baseline。
2. 真正接入 `G05PolicyQwen35` 训练和生成状态：旧 `generate_text` 的位置状态更新、BAR EOS/PAD、真实生成的代码提交、取消未执行计划、FM 路径保持隔离。验证原生多位置 CE 和真实 ActionCodec；BAR16 通过新训练配置显式开启。
3. GPU FP32/BF16、FLA / CUDA convolution、FlashAttention、Liger CE 一致性，原始单轮回归，padding/变长 batch/packing，真实多轮数据监督。当前 native API 不接受 padding；不能把这个限制误称为完整的 episode packing。
4. 真实 Codec 粒度和联合分布：比较 AR 1/8/16/32、原生 RVQ 结构及语义分组，避免将 V4 toy 的锚点策略直接硬编码到真实 codec。
5. 相同真实基座、数据及训练预算下评测 LIBERO、RoboTwin、历史/ICL、干扰与连续控制；真实环境验证后才能得出性能优越性结论。

**精简原则：只扩展既有接口、修正已有计算语义、复用原生编码器和共享 CE。不以大量 Toy 实验替代真实性能验证。**
