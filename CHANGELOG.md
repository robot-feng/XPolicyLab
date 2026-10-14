# Changelog

## Unreleased

- Add optional task constraints, explicit trial states, one-run ownership and
  idempotent host stop requests; preserve blocked, interrupted and unstarted evidence.
- Complete native episode reset before offline READY and capture inference timing,
  control-tick queue delay, local receipt age and optional process CPU accounting.
- Add results schema 2 with evidence hashes and verified summaries that preserve
  failures, judge sources and denominators while pooling comparable raw samples.

- Add explicit StarVLA raw/normalized state precision, handshake checks and
  launcher forwarding; keep float32 defaults and reject nonfinite/overflowing states.
- Add a genuine G2 model-boundary diagnostic using the original loader, checkpoint,
  actual XPL normalization wrapper and optional CUDA inference with a memory cap.
- Revise the v4 blueprint from implementation evidence: distinguish packed/final
  model boundaries, returned/planning horizons and model/device acceptance scope.

- Add the `xpolicy` import and CLI alongside the existing XPolicyLab package.
- Add strict, versioned model/robot/dataset profiles and canonical RGB observations.
- Validate camera names, state/action ordering, physical units and robot compatibility.
- Add lazy policy discovery, a deterministic CPU mock and explicit legacy/chunk runtimes.
- Reuse the existing WebSocket RPC for the new policy service; preserve legacy outputs.
- Verify checkpoint, loader and statistics identities before importing model adapters.
- Manage project dependencies through `pyproject.toml` with uv in existing Conda environments;
  provide a separate core lock and CPU CI for reproducible installation.
- Add protocol and contract checks plus evidence reports that retain unfinished checks.
- Add explicit HDF5 dataset mappings and synchronous, noncausal `xpolicy eval` replay.
- Record full predictions, execution prefixes and per-field behavior cloning errors;
  preserve recorded observations and leave physical feedback/task success unmeasured.
- Export genuine local G2 training DataLoader samples with source/weight identities,
  lossless shared-decoder roundtrips, padding masks and named statistics comparisons.
- Add hash-verified golden bundles and `xpolicy golden diff` for first-boundary diagnosis.
- Add an explicit StarVLA PIL resize option and verify its image preprocessing against
  genuine loader samples; preserve the existing OpenCV resize default.
- Add a shared record-only chunk buffer, generation-aware latest-only inference,
  explicit latency trimming and absolute-position overlap with contributor traces.
- Add paced asynchronous dataset replay and actual WebSocket reset interleaving checks;
  retain first faults, cleanup errors, skipped frames and unmeasured feedback in reports.

## 72f6b36f

- Add the shared versioned `ActionChunk` and optional complete-chunk model/client API.
- Add StarVLA complete predictions without advancing its legacy execution cursor.

The architecture blueprint remains a design specification. EVA migration is deferred;
dataset wire runtime, final model tensor/state parity and device acceptance remain outstanding.
