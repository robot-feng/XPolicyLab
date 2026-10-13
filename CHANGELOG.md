# Changelog

## Unreleased

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

## 72f6b36f

- Add the shared versioned `ActionChunk` and optional complete-chunk model/client API.
- Add StarVLA complete predictions without advancing its legacy execution cursor.

The architecture blueprint remains a design specification. EVA migration is deferred;
dataset wire runtime, genuine training-loader parity and device acceptance remain outstanding.
