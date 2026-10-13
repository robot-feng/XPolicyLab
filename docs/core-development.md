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

## Evidence and remaining work

`tools/validate_core.py` writes `TEST_REPORT.json`, exact commands, exit codes, log paths and
hashes, source-file hashes and a software inventory. Reports retain `FAIL`, `BLOCKED` and
`NOT_RUN` entries. `implementation_status` covers only implemented core commands;
`blueprint_status` stays `INCOMPLETE`. A contract-mode `PASS` is scoped to declared contracts;
`environment_runtime` remains `NOT_RUN`.

The current experiment fixture names `fake_plant` for preflight, but no device or fake plant is
started by doctor. Dataset direct/wire replay, genuine training-loader golden samples,
normalization parity, generation-aware execution, SDK mappings, shadow and physical safety
acceptance are still pending. EVA migration is deferred by user instruction. See
[Migration status](migration-status.md) for upstream provenance and acceptance scope.
