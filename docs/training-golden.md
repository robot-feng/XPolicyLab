# Training-source golden samples

The G2 exporter runs the local StarVLA training registry's actual
`PaddingMaskG2Dataset` through a PyTorch `DataLoader`. It records raw RGB/state/action
data at O1 and the packed training sample at O2: PIL-resized images, float16 normalized
state/actions, language and action padding masks. O2 here is **before the VLM processor**;
this does not establish final model tensor or checkpoint forward parity.

## Export in the existing model environment

Keep Conda's Python/PyTorch/CUDA and install this project with uv:

```bash
conda activate XPolicyLab
uv pip install --python "$CONDA_PREFIX/bin/python" -e .
python tools/export_g2_training_golden.py \
  --source /path/to/starVLA-RollFlow \
  --dataset /path/to/AgiBotG2/white_carton_to_left_tan_carton_20261009 \
  --checkpoint /path/to/g2-training-run \
  --output runs/g2-training-golden/baseline
```

The exporter requires the checkpoint's `config.full.yaml`, `config.yaml`,
`dataset_statistics.json` and `final_model/pytorch_model.pt`. It verifies dataset/mix,
absolute actions, state inclusion, dimensions and declared group concatenation. It
selects the first training episode's first, middle and last steps, retaining the actual
parquet timestamps rather than replacing them with nominal `frame / fps`.

Metadata is copied to a new output directory; the data/videos directories are linked
for reads. Loader caches are written only to that copy. Original source/config/metadata,
sample parquet/video and weight hashes are checked for stability. The report includes
current training source hashes, git identity, seed, loader configuration and an environment
inventory. The checkpoint's historical training source revision is not known from these
files. Do not equate the current checkout with that historical implementation.

Each sample directory contains:

| Artifact | Evidence |
| --- | --- |
| `training/` | Genuine O1/O2 manifest and numeric NPZ arrays |
| `raw-sample.hdf5` | Lossless raw pixel/state/label sample; not a complete evaluation trajectory |
| `hdf5-roundtrip/`, `roundtrip-diff.json` | O1 recovery through the shared RGB decoder |
| `opencv-resize-counterexample/`, `resize-diff.json` | Isolated INTER_AREA counterexample; other training arrays are copied to isolate resize |
| `training-images/`, `adapter-pillow-images/`, `adapter-image-diff.json` | Image-only comparison against the actual `Model._convert_obs` method |
| `TRAINING_GOLDEN_REPORT.json` in the run root | Export status, statistics alignment, hashes, limitations and unexecuted checks |

The raw images are stored as decoded RGB arrays. Re-encoding as JPEG would introduce
loss and invalidate an exact pixel comparison. No additional image decoder or channel
conversion is used. The standalone sample HDF5 is not an `xpolicy eval` dataset: it
contains a single observation and a full label horizon, and unit metadata is absent.

## First divergence diagnosis

```bash
xpolicy golden diff \
  --expected runs/g2-training-golden/baseline/episode-000001-frame-000000/training \
  --observed runs/g2-training-golden/baseline/episode-000001-frame-000000/hdf5-roundtrip \
  --layers O1 --output runs/g2-training-golden/o1-diff.json
```

The exit code is 0 for a comparison PASS and 2 for a difference or invalid bundle.
Both NPZ and individual array hashes are verified; pickle/object arrays, nonfinite
values, duplicate JSON keys and invalid axis labels are rejected. Comparison checks
sample/frame/clock identity, ordered semantic metadata, array names, shape, dtype and
axis labels before numeric values. It reports the earliest divergent layer, field,
index and named joint where available. O1 precedes O2, followed by A0–A5.

Float tolerance defaults to zero; `--atol` and `--rtol` explicitly relax float
comparisons. Integer RGB pixels remain exact at every tolerance. Selecting `--layers`
records omitted layers in the report. A PASS covers the selected diagnostic boundaries;
it does not verify physical calibration, model forward behavior or device operation.

## Recorded local findings

The local G2 corpus has 109 episodes / 27,233 frames and 24 named state/action fields:
left arm, left inner gripper, right arm, right inner gripper, head and waist. The audited
checkpoint declares a 64-step action horizon and state inclusion in `config.full.yaml`.
The training split has 24,913 steps. The first selected episode is episode 1, with
sample frames 0, 129 and 257; its last sample has 63 padded actions.

All three O1 lossless roundtrips pass exactly. The legacy OpenCV AREA resize differs
from the training PIL resize at O2; the first head image differs in 56,892 channel
values. With `image_resize_backend: pillow_bicubic`, the actual adapter image method
matches all three cameras in all three samples exactly in the recorded Conda environment.
The default remains `opencv_area` for existing deployments. Select the backend from
the checkpoint's genuine preprocessing contract, not from tensor dimensions alone.

Checkpoint q01/q99 vectors match the current loader's named order exactly. Two std
vectors differ by less than `9e-14`; the report retains exact-match failures separately
from a declared numerical tolerance (`atol=1e-12`, `rtol=1e-7`). This does not verify
deployment normalization frequency or checkpoint inference.

Source unit metadata is absent and is recorded as null. No G2 hardware RobotProfile
or inner/outer gripper mapping is approved by this export. The existing arm/gripper
adapter does not establish a full G2 head/waist mapping. Model processor/state parity,
checkpoint forward, dataset wire, device mappings and on-site acceptance remain pending.
EVA migration remains deferred.
