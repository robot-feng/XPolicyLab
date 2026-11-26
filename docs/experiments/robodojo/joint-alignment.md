# RoboDojo joint 数据版本与时序验收

2026-11-26 核对 KnowAct 和 G0.5 正式对比的数据来源。官方于 9 月 15 日在原目录覆盖 LeRobot v3.0 joint 数据，每条轨迹去掉最后一行状态和动作，并去掉第一帧图像。两组共同指定提交 `abdcac560cf05782938f69a9e6f6717846f46b33`，目录为 `data/RoboDojo_lerobot_v30_video`。本次抽样通过文件版本和关节列校验，episode 0 的多相机时序验收失败。正式对比待数据验收通过后启动。[官方修复说明](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/discussions/42) · [固定版本目录](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/abdcac560cf05782938f69a9e6f6717846f46b33/data/RoboDojo_lerobot_v30_video)

本机两个训练入口均读取 `ROBODOJO_LEROBOT_V30_ROOT`。两份 `configs/data/robodojo.yaml` 内容相同，三个相机、状态和动作的 `time_offset` 都是 0。核查时环境变量为空，工作区、数据盘及 Hugging Face 缓存中未找到可追溯的完整 joint v3 快照。此次单独下载了抽查用的 6 个 LeRobot 文件和 2 个同提交的官方原生 HDF5，逐一核对了官方内容标识。本机保留于 `runs/robodojo/alignment-audit/<提交号>/`。完整 joint 目录约 120 GB，本次本地样本只用于诊断。[G0.5 训练入口](../../../policy/G05/train.sh) · [KnowAct 训练入口](../../../policy/KnowAct/train.sh) · [样本文件与 SHA-256](episode0.sha256)

## 逐帧结果

抽取 `arrange_largest_number/arx_x5` 的 episode 0。原生 HDF5 带 `fix_obs_shift_rgb=shift_drop_first_drop_last+rgb_jpeg` 标记。Parquet 的 578 行帧索引和 25 Hz 时间戳连续；每行 14 维状态和 14 维动作按左臂、左夹爪、右臂、右夹爪拼接，均与原生 HDF5 转成 float32 后逐元素相同。

按训练读取器的 `from_timestamp + timestamp` 在 MP4 中取帧。用共享函数解码 HDF5 中的图像，把两种导出都缩小到 160×120、按 RGB 三通道取均值，比较亮度绝对误差（像素量程 0–255）。此方法检验时序。下表平均 576 个内部时刻，偏移是相对于 MP4 帧序号的 HDF5 帧序号，数值越小越匹配。[v3 读取实现](../../../policy/KnowAct/G05/src/g05/data/lerobot/lerobot_dataset_v3.py)

| MP4 相机 | HDF5 前一帧 | 同帧 | 后一帧 | 按元数据计算的该回合视频帧数 |
|---|---:|---:|---:|---:|
| 顶视 | 4.54 | 3.09 | **1.11** | 577 |
| 左腕 | 6.47 | 4.39 | **1.45** | 577 |
| 右腕 | 4.74 | **1.19** | 4.73 | 578 |

按原生前后两帧之间的变化量，在每个相机独立挑出运动最大的 144 个时刻，再比较同帧和后一帧。顶视为 7.32 和 1.08，左腕为 10.44 和 1.97，右腕为 1.26 和 11.51。运动明显的片段同样支持相机间的时序差异。作为正对照，将原生顶视第 200–219 帧按原顺序无损编码后交给同一比对函数，前一帧、同帧和后一帧的平均误差分别为 6.43、0.00、6.01。

第 577 行（从 0 起算）对应 23.08 秒，也是顶视、左腕在元数据中标出的下一回合起点。对照两个原生回合，这一时刻的顶视和左腕与 **episode 1 首帧**的误差为 0.99、1.23，与 **episode 0 末帧**的误差为 3.78、3.15。右腕对应值为 4.47、1.19。另用训练环境默认的 TorchCodec 解码每路相机第 17、119、265、410、523、577 行，所得 18 张 uint8 RGB 与 PyAV 逐像素一致。

对整张 episode 元数据表计算 `round((to_timestamp - from_timestamp) * fps) - length`。3,500 条轨迹合计 1,856,102 行；顶视有 23 条短一帧，左腕 14 条，右腕 6 条，去重共 38 条。其余条目时长与行数相符。逐帧验证的范围是 episode 0 和相邻回合首帧，其他回合的视觉时序还需按内容核对。两种官方导出的比较也无法单独还原模拟器各传感器的原始采样时刻。

## 复现和正式对比条件

在仓库根目录执行。以下命令按提交号下载上述样本到独立目录，使用项目现有 KnowAct Python（PyAV、OpenCV、PyArrow、h5py）。校验脚本常规返回 0；图像比对或回合长度检查失败时返回非零，本次固定数据的结果为 `episode_passed: false`、退出码 1。

```bash
REV=abdcac560cf05782938f69a9e6f6717846f46b33
REPO=RoboDojo-Benchmark/RoboDojo
SAMPLE="$PWD/runs/robodojo/reproduce/$REV"
SUMS="$PWD/docs/experiments/robodojo/episode0.sha256"
mapfile -t FILES < <(awk '{print $2}' "$SUMS")
hf download "$REPO" "${FILES[@]}" --repo-type dataset --revision "$REV" --local-dir "$SAMPLE"
(cd "$SAMPLE" && sha256sum -c "$SUMS")
RAW="$SAMPLE/data/RoboDojo/arrange_largest_number/arx_x5/data"
PYTHONPATH="$PWD/..:${PYTHONPATH:-}" policy/KnowAct/.venv/bin/python \
  scripts/RoboDojo/audit_joint_episode.py "$SAMPLE/data/RoboDojo_lerobot_v30_video" \
  "$RAW/episode_0000000.hdf5" --episode1 "$RAW/episode_0000001.hdf5"
```

准备完整对照时，两组使用同一只读源目录，并记录同一份训练/验证 episode 清单、快照提交及派生文件哈希。完整上游快照可用以下命令固定；`hf download` 返回含提交号的快照路径。旧下载脚本可根据已存在的 `.download_complete` 直接跳过下载，因此旧目录的文件名和完成标记不足以验证版本。

```bash
SNAPSHOT=$(hf download "$REPO" --repo-type dataset --revision "$REV" \
  --include 'data/RoboDojo_lerobot_v30_video/**' --quiet)
export ROBODOJO_LEROBOT_V30_ROOT="$SNAPSHOT/data/RoboDojo_lerobot_v30_video"
```

两组均保持零额外偏移。后续优先复查一个时长相符的回合和一个异常回合，定位打包差异；如需从同提交原生 HDF5 重导出，另建共享数据目录，验证回合首尾、运动段及跨回合边界，再固定双方使用的文件清单。这里核查的是 RoboDojo 双臂数据；此前 PatternLock 训练使用 RoboMME 原生 Panda 数据，其时序与闭环结果见独立实验记录。[RoboMME 实验记录](../knowact-robomme/fit-ladder.md)
