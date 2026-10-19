# Changelog

## Unreleased

- Clarify that `EVAL_ENV_TYPE=real` selects an environment-side client and does not imply
  a bundled generic hardware driver; revise the v4 roadmap to reuse existing clients and
  consider EVA only for a verified capability gap.
- Add a staged RobotBackend/EVA integration plan with process-boundary checks, observation
  and action goldens, protocol/execution fault injection, Fake/SDK Mock, zero-actuation
  Shadow and manually gated controlled motion; planned checks are not reported as PASS.
- Route ME-Dex Clean50 HDF5 camera frames through the shared RGB decoder so raw arrays,
  legacy RoboTwin JPEGs and marked JPEGs have one channel contract; add layout-specific
  regression checks and document the released checkpoint's separate BGR inference setting.
- Fix TinyVLA's two HDF5 loaders inspecting/resizing encoded buffers before decoding
  and swapping RGB channels after decode; use one shared-codec image boundary.
- Preserve raw-image resize behavior and add real Torch/HDF5 method checks plus a
  retained Git-baseline diagnostic; checkpoint training/evaluation remains unverified.

- Add explicit configuration/base-seed suites using separate sequential Runner processes,
  preserving blocked, failed, timed-out and unstarted cases without inventing trial scores.
- Bind child results to their planned config/profile/seed identities; verify closed suite
  snapshots, process receipts, nested evidence, source stability, denominators and summary.
- Allow an explicitly selected compatible case interpreter, preserving venv executable
  links, pinning its identity and checking shared Runner source; record actual runtime inventories.
- Share owned process-group supervision with SDK probes; record native exits and bounded
  host cleanup, with SIGINT/SIGTERM suite stopping and no physical stop claim.
- Add supervised actual SDK startup probes and installed native-stack constraints; retain
  measured Isaac package/platform/hardware blockers and restore its original SDK inventory.

- Introduce lazy environment capability/status discovery and the recorded dataset
  provider, owning HDF5/record-only sink/context with generation-aware reset and stop.
- Route Runner timeline access/submissions through provider receipts, trace provider
  preparation/reset and verify descriptors in report summaries; separate trial evidence code.
- Add `xpolicy environments list` and a provider preflight doctor mode; preserve the
  default contract-only diagnosis and explicitly defer unavailable simulator/device modes.

- Add `xpolicy assets inventory` for tracked vendor/license/resource discovery and
  pinned external asset declarations, with explicit scope and unresolved findings.
- Record safe internal license-link bindings while refusing external/chained links;
  distinguish identity verification and documented evidence from licensing permission.

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
dataset wire runtime, complete registered G2 runtime and device acceptance remain outstanding.
