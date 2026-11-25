> **版本说明（V7.1 整理版）：** 本文件是 V7 阶段原始开发报告，文中的 69/69 是 V7 当时的测试数。整理版修正了比较脚本的 G05 默认 action source（官方 RoboDojo G05 默认 FM），新增 paired comparison 测试，并重新运行 73 项 CPU 回归测试。当前代码与复现方式以 `README.md` 和 `docs/` 为准。

# KnowAct V7 — XPolicyLab

## Decision and scope

Continue the supplied XPolicyLab snapshot 4a9e3afd73b86a48c5db076f5cf9dbfbef1a180e from V6, only modify policy/KnowAct. Original G05 stays untouched. No new trainable parameters; native Qwen3.5 model and GDN files byte-identical to V6. Same CE, same shared LM head, same original ActionCodec; BAR16 remains default for newly trained models. CPU PyTorch 2.10, no GPU, official weights/processor/codec and real LeRobot dataset unavailable. Hugging Face DNS access failed. All numerical training results below use the actual small G05 source architecture, test text mapping, randomly initialized small FSQ/rule codec, synthetic observations/actions. They are not pretrained G05 comparisons. Prior V6 report and artifacts retained for provenance. No trained V7 checkpoint is distributed; test runs record metrics and seeds.

## A. Diagnose low CE versus free generation

Three seeds (11,22,33), two fixed repeated in-distribution cases, same current observation with distinct historical command. Original BAR labels checked position-by-position; actual full and incremental hidden states max absolute difference < 6e-6. Variable-token accuracy excludes markers and PAD. 100 updates vs final checkpoints 200/300/300 updates. This is an optimization diagnosis, not an independent test or matched-budget comparison.

| Metric | 100 steps | extended |
|---|---:|---:|
| variable-code teacher forcing |89.22%|99.02%|
| variable-code free generation |77.94%|97.06%|
| gripper exact rule |16.67%|100%|
| whole action sequence exact |0%|66.67%|

Two seeds obtain both exact cases; seed22 still fails exact match. Not evidence for a new model module. Diagnostic aggregate CE includes contextual labels, so do not interpret it as the training objective. Raw per-slot errors, IDs and losses: results/diagnosis*.json.

## B. Native Context Learning: held-out demonstration permutations

Three valid gripper trajectories, three symbols, six possible bijections. Four mappings in training, two held out. Each sample's three demonstrations have random order; current observations do not disclose mapping. Historical actions are four applied steps, then hold-fill to eight-step codec horizon. 240 optimizer updates per run, two episodes per update, three seeds x three modes: current only, native history, native history + optional fixed-format target masking. All use same backbone and CE, not identical compute/token budgets; masking is an experimental switch, OFF by default. Evaluation 12 seen and 18 held-out cases per seed, repeated orders of small finite mappings; no independent real-robot test.

| Mode | Held-out variable code | Gripper | Entire action | Same model without history: gripper |
|---|---:|---:|---:|---:|
| current |90.03%|22.22%|11.11%|22.22%|
| history |96.08%|33.33%|33.33%|33.33%|
| history + mask |92.16%|22.22%|11.11%|22.22%|

History model seen gripper 33.33%, held-out and history reset 33.33% for all seeds. Random-choice reference 1/3. Does not establish ICL. Repeated mostly-identical motion codes dominate aggregate accuracy, so must measure decisive action fields and intervention. Mask did not show a benefit in this setup. An earlier invalid third gripper trajectory violated the original continuity rule and was automatically repaired differently in demonstration/target. Aborted and excluded; evidence preserved under results/aborted_invalid_grip. Corrected trajectory validity asserted in experiment_v7.py. All nine final run JSONs include per-case outcomes. No further architectural addition justified. Training on actual pretrained VLM/codec, broad language/demonstration data and task-level outcomes remains necessary.

## C. Structured Action Autoregression

Actual original rule tokenizer with horizon32, base4096, min_block_len1: 7,049,156 valid encodings out of 16,777,216 possible two-code pairs. For constant closed and constant open trajectories their codes are [0,0] and [1720,4035]. Both mixed pairs are grammar-valid. Equal independent marginals yield 100% grammar validity, 50% intended-mode mass. Thus grammar constraints alone cannot ensure intended motion. This is exact rule-code enumeration/case analysis, NOT observed G05 model error frequency.

Actual small random FSQ codec: 192 synthetic paired alternatives (64 x 3 seeds). Excluding gripper variation, all distinguishable. At the earliest differing block with d differing positions, the product of exact equal-marginal binary distributions assigns mass 2^(1-d) to the two intended combinations. Average diagnostic across contexts: width1=100%, width8=3.6377%, width16=0.06574%, width32=0.00005394%. This is a constructed binary-mixture calculation, NOT measured free-running accuracy, nor an estimate for pretrained RVQ. Does not prove grouping or changing width will help on real data. Added scripts/audit_action_codes.py for legitimate paired alternatives encoded by the actual target checkpoint. No new ordering promoted; compare codec-aware ordering and same BAR16 baseline only when dependencies on real expert alternatives are measured.

## D. Continuous closed-loop interaction

V6 could carry a full 32-step teacher plan into a later decision after 16 steps, including future labels. V7 task config now defaults to 16-step training stride and dispatch. For historical turns take only actually available expert command prefix, repeat final available command to the fixed codec horizon, mark suffix as padded, and set entire historical turn context-only. The current action target remains full horizon. Runtime generates a full action plan through native generate_action, keeps only the pre-action committed cache, and stores normalized proposal until receipt. At next observation, re-encode precisely acknowledged prefix + deterministic hold-fill and then append new measured observation with receipt. Zero applied steps discards proposal; stale, missing, invalid or duplicate receipt rejected; failed decode/preprocess cannot commit candidate. Hold-fill is for representation only, never sent for physical execution. Actual measured state is supplied separately; applied-step receipt confirms commands applied, not measured trajectory accuracy. No async mid-kernel interruption.

Training preprocessing support limited to verified RoboDojo joint configuration: equal data/control FPS, no future obs or action offsets, single current observation, no extra action transforms, DummyActionFilter, identical pointwise VLM/action normalization (no stepwise norm). Unsupported transforms rejected. Runtime derives actual pad mask through native GroupedPaddingMerger using shape-only zero tensors; physical 14-dimensional joint action is represented in 20-dimensional layout, padding [6,7,8,16,17,18] zeroed in history. Legacy checkpoints without native_replan_steps keep V6 semantics. Reset ablation uses same checkpoint without history. Explicit stride mismatch allowed only as labeled stress test. No automatic memory eviction; context budget hard limit.

Original codec decoder is not made causal by this change. Perturb one late legal neural code in each of 192 small-configuration samples; all 192 have nonzero change in first four decoded steps, max 0.004220843. Therefore execute only from a fully decoded chunk; no claim of safe partial BAR decoding. Real pretrained codec still requires its own prefix perturbation test. The historical record can be causally re-encoded even when the codec itself mixes time. Need train on actual closed-loop corrections/perturbations before claiming execution recovery, and validate decoded command error.

## E. Tests and benchmark readiness

69/69 CPU regression tests passed (previous 54 + 15). Includes actual G05Policy/ActionCodec, replay equality for 1,3,4,7,8 applied steps, future suffix interventions, zero receipt, transaction/cache isolation, action shape padding, mask gradient, XPolicyLab adapter with simulated IO, paired launch and scoring. Original G05 untouched. Test text mapping and optional dependency stubs, simulated IO; no real checkpoint/GPU/LeRobot/RoboDojo test. See results/v7_final_regression.log and v7_preflight.json. Training validation loader currently uses current-only original evaluation path; paired full-history validation requires real data evaluation, not assumed. Formal compare.py now supports G05_MEM and a separately trained KnowAct_current baseline, plus same-checkpoint reset; no automated success extraction or benchmark scores fabricated. Require real matched task instances, seeds, budgets and execution intervals; report actual observation offsets and history configuration, paired success, latency, memory and failure reasons. No claim V7 outperforms G05. 

## F. Acceptance gates

1. Load official model, processor and codec; verify missing/unexpected checkpoint keys, original output with disabled context, single/multi-turn BF16 parity.
2. Load real LeRobot episode and check target labels, physical/normalized action alignment, strict split and receipt semantics. Reproduce matched 16/32 training-dispatch.
3. Establish pretrained G05 (including native memory) baselines, same-budget single-turn BAR16 and native-context weights, and reset ablation. Measure actual held-out demonstrations, current-vs-history interventions, decisive action fields, exact and continuous reconstruction; don't select by CE alone.
4. Audit real codec paired expert alternatives and prefix causality. Only promote new generation ordering after matched budget empirical benefit. No new model head. If genuine mid-chunk streaming is necessary and pretrained codec fails causality, train/test a causal codec separately; no claim that V7 already supports this.
5. Closed-loop perturbation test with measured state changes, interruptions, and actual task success. Do not proceed to physical robot based only on synthetic tests.
