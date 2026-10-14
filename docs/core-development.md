# Core development

## Conda provides the environment; uv manages Python packages

Use a compatible existing Conda environment, or clone one with the model stack already installed:

```bash
conda create --name XPolicyLab --clone <existing_env>
conda activate XPolicyLab
uv pip install --python "$CONDA_PREFIX/bin/python" -e .
xpolicy --help
xpolicy doctor --config configs/experiments/mock_contract.yaml
```

Python dependencies are declared once in `pyproject.toml`. The core does not require PyTorch,
CUDA, ROS or a vendor SDK. `uv pip install` preserves installed packages satisfying these
requirements and installs missing dependencies. Use the activated environment's `python` and
`xpolicy` commands. See uv's [environment discovery](https://docs.astral.sh/uv/pip/environments/)
and [package installation](https://docs.astral.sh/uv/pip/packages/) documentation.

For development checks, install the optional extra in the same environment:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e '.[test]'
python tools/validate_core.py --environment conda --output runs/core-validation/conda
```

The report records the whole environment's dependency check. A cloned model environment may
contain conflicts among model-specific packages even when core dependencies satisfy their
requirements. Retain those failures in the report; handle them in that policy's environment.

### Preserve an installed native stack during resolution

The package probe can run another Conda interpreter without installing the public core
there. It writes installed native versions as constraints, including NumPy, Torch,
CUDA wheels and simulator packages. A constraint limits resolution; it does not request
installation of that package. Dependencies continue to come from `pyproject.toml`.

From the core environment, after cloning the intended environment:

```bash
python tools/probe_simulator_runtime.py \
  --python /path/to/cloned/env/bin/python --mode packages \
  --output runs/cloned-env-packages
uv pip install --dry-run --python /path/to/cloned/env/bin/python \
  --constraints runs/cloned-env-packages/native-stack-constraints.txt -e .
uv pip install --python /path/to/cloned/env/bin/python \
  --constraints runs/cloned-env-packages/native-stack-constraints.txt -e .
uv pip check --python /path/to/cloned/env/bin/python
```

Inspect the dry run before installation. The constraint snapshot is a local environment
input, not another project dependency declaration or the public lock. It cannot establish
whole-environment compatibility or prevent every transitive change. Retain package inventories,
the resolver plan, installation log and dependency-check failures. Use a new probe output
directory for each attempt.

An actual Isaac Sim 5.1 kernel inspection found exact `websockets==12.0` and
`coverage==7.4.4` requirements. The public core requires WebSocket 14–15, so installing
both in one interpreter creates an incompatible environment. Keep that SDK in its own
interpreter and run the supervisor/core separately. A local installation attempt was
recorded and reversed: all SDK package names and versions again match the pre-install
inventory. This is evidence for process separation, not permission to ignore dependency
errors. See [simulator runtime evidence](environments.md#local-runtime-evidence).

Use exact `uv sync` only in the dedicated core environment: it removes undeclared packages
by default. uv documents this behavior in [sync semantics](https://docs.astral.sh/uv/reference/cli/#uv-sync).

## Locked CPU installation

`uv.lock` locks public core dependencies and the test extra. It does not replace a policy's
Conda/CUDA/SDK requirements. CI uses a clean `.venv`:

```bash
uv sync --locked --extra test --default-index https://pypi.org/simple
uv run --no-sync python tools/validate_core.py --environment cpu --output runs/core-validation/cpu
uv run --no-sync python tools/validate_install.py --output runs/core-validation/wheel
```

This is a separate installation check. In an existing Conda model environment, use the
`uv pip install` workflow above. Lock regeneration is explicit:

```bash
uv lock --default-index https://pypi.org/simple
```

## CLI and profiles

```bash
xpolicy policy list
xpolicy policy serve --config configs/policies/mock.yaml --port 19000
xpolicy doctor --mode contract --config configs/experiments/mock_contract.yaml \
  --output runs/mock-contract/doctor.json
```

Configuration paths resolve relative to their declaring YAML file. Unknown fields, duplicate
YAML keys, placeholders, unsupported schema versions and disabled contract checks are errors.
`configs/robots/mock_arm.yaml` and `configs/policies/mock_manifest.yaml` describe a synthetic
two-field mock fixture. They do not register a physical robot or define G2 hardware semantics.

`ModelManifest`, `RobotProfile` and optional `DatasetProfile` are compared before policy imports.
Matching dimensions alone are insufficient: action type, named field order, units, reference
frame, quaternion ordering and required cameras must agree. Additional camera channels are
allowed; required cameras cannot be silently replaced with black images. Boundary images are
HWC uint8 RGB. Existing `decode_obs_images` owns wire decoding; adapters do not decode or swap
channels.

The canonical observation carries state names/units, source frame, episode, timestamp and clock
domain. Arrays are copied and made read-only. Profiles declare raw physical public inputs and
outputs; any checkpoint normalization belongs inside the policy. Manifest declarations do not
prove that a genuine training loader and policy processor produce equal tensors.

## Policy API ownership

Policy discovery stores import paths; listing policies does not import their models. Existing
adapters remain under `policy/<NAME>/`. Their installation and benchmark scripts still use the
parent workspace's `env_cfg/` and shared checkpoint/action-dimension helpers.

Choose one API when constructing a service:

| API | Request | Result | Owner of chunk consumption |
| --- | --- | --- | --- |
| `legacy` | existing `INFER` / `CALL get_action` | original adapter output | original adapter |
| `chunk` | `CALL predict_action_chunk` | full typed `[T,D]` prediction | caller/runtime |

Legacy services reject the complete-chunk API; chunk services reject legacy inference and action
calls. The mock exposes only `chunk`; StarVLA declares both capabilities, with one selected per
service. No legacy single step is re-labelled as a full horizon. Chunk validation checks the
manifest horizon, dimensions, physical fields and request context. An episode change requires
`RESET`. `execution_ready` reports metadata completeness only; it does not check expiry, device
bounds, calibration or permission to move. The mock does not invent a validity deadline.

## Artifact identities

Checkpoint manifests require checkpoint and training-loader identities. Normalized checkpoints
also require a statistics identity. Materialize these artifacts locally and declare their paths
relative to the manifest. The policy factory checks their hashes before importing vendor code;
an explicit `policy.checkpoint` must equal the verified checkpoint path. `doctor
--verify-artifacts` performs the same content checks; ordinary contract mode leaves them
`NOT_RUN`.

File identity is SHA256 of file bytes. Directory identity uses SHA256 initialized with
`XPL-DIR-SHA256-v1\n`. For every regular file, sorted by POSIX relative filename, append the
UTF-8 filename length as eight unsigned big-endian bytes, the filename bytes and the raw
32-byte SHA256 of the file. Empty directories are ignored; symlinks and special files are
rejected. Compute an identity with `xpolicy.core.artifacts.artifact_sha256(Path(...))`.

## Dataset direct replay

```bash
python tools/create_mock_dataset.py --output runs/mock-replay-fixture
xpolicy doctor --config runs/mock-replay-fixture/experiment.yaml --verify-artifacts
xpolicy eval --config runs/mock-replay-fixture/experiment.yaml
```

The fixture generator requires a new output directory and declares `source_kind: synthetic`.
For a recorded trajectory, supply a `DatasetProfile` with `decoder_id: xpl_hdf5_v1`, a verified
HDF5 artifact, and explicit mappings. Every camera has its HDF5 `source_key`. Every state field
maps to `{key: state/<array>, column: <index>}` under `hdf5.state`; optional action labels use
the same structure under `hdf5.action`. These mappings select columns; they never infer a
robot's physical order from vector length. Units are the declared physical units, with no
implicit conversion or normalization. Duplicate source-column bindings are rejected.

Camera/state/action streams must have equal lengths. Optional `timestamp_key` and
`frame_id_key` identify strictly increasing recorded arrays. Without them, time is explicitly
reported as `nominal_index_rate` and frame IDs are row indices. `instruction_mode: constant`
reads a scalar string or JSON list of alternatives, selecting `instruction_index`;
`per_frame` requires one string per source row. Camera bytes decode through the existing
`decode_image_bit`; encoded, legacy and already-decoded layouts retain its RGB semantics.

The runner currently accepts only `dataset_direct`, `record_only`, `sync`, and a complete-chunk
policy. One HDF5 file is one episode (`episode: 0`); the robot/control/dataset rates must agree.
It predicts a full chunk, records the `execute_horizon` prefix, then advances that many
recorded source rows. The final prefix is truncated to the remaining rows. RecordOnly has
no actuator and cannot update recorded state or images. Frame seeking uses source row indices;
recorded frame IDs are preserved in every observation and trace.

Each run gets a unique directory with `config.json`, `profiles.json`, software inventory,
`trace.jsonl`, optional decoded RGB/state NPZ inputs, and `results.json`. Trace events link
run/trial/episode/frame/request/chunk/sequence IDs. Each repeat resets policy state. Action
labels are compared by physical field and unit, reporting MAE/RMSE as behavior cloning error.
Label disagreement is not a protocol fault. Task success and device feedback remain unmeasured.
Model exceptions and elapsed-budget overruns produce `FAULT` trials, retaining available
evidence. Duration is checked between synchronous predictions and when they return; this
runner does not interrupt a blocked model call. The seed is recorded and NumPy is seeded;
backend-specific randomness still needs that policy's own seed configuration.

## Evidence and remaining work

`tools/validate_core.py` writes `TEST_REPORT.json`, exact commands, exit codes, log paths and
hashes, source-file hashes and a software inventory. Reports retain `FAIL`, `BLOCKED` and
`NOT_RUN` entries. `implementation_status` covers only implemented core commands;
`blueprint_status` stays `INCOMPLETE`. A contract-mode `PASS` is scoped to declared contracts;
`environment_runtime` remains `NOT_RUN`.

The current experiment fixture names `fake_plant` for preflight, but no device or fake plant is
started by doctor. [Training-source golden samples](training-golden.md) now export the actual
G2 DataLoader and compare lossless raw data plus adapter image preprocessing. Final model tensor/
normalization parity, dataset wire replay, SDK mappings, shadow and physical safety
acceptance are still pending. EVA migration is deferred by user instruction. See
[Migration status](migration-status.md) for upstream provenance and acceptance scope.

[Record-only chunk execution](execution.md) documents the shared sync/async buffer,
generation resets, latency trimming, overlap algorithm, paced replay and timeout limits.
