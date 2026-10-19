# Migration status and provenance

## Scope

This checkout is the `robot-feng/XPolicyLab` fork. The v4 blueprint is a design and
acceptance specification; the full unified platform is not implemented. The current
change implements the public core, policy service, offline diagnostics and record-only
execution foundations. EVA changes were
deferred by the user. No EVA robot, transport, SDK, recorder or source archive is
imported, modified or bundled in this change.

The README's `EVAL_ENV_TYPE=real` selects an environment-side client when its hardware
integration exists. The policy adapter and RPC do not constitute a generic hardware
driver or safety controller; an existing client may be sufficient, so EVA is considered
only against a concrete, measured capability gap.
The staged integration gates and fault-injection plan are documented in
[Robot Backend integration](robot-backend-integration.md); those planned checks are not
current runtime evidence.

## Source provenance

| Component | Origin / revision | Treatment |
| --- | --- | --- |
| XPolicyLab baseline | `https://github.com/XPolicyLab/XPolicyLab`, `f6e309e8ece20bfa33c0e5d7b05a5f3287522757` | Apache-2.0 license retained; NOTICE added |
| Shared ActionChunk and StarVLA chunk entry | Local commit `72f6b36f985cd219e35b48441d1ea825c71cc80b` | Reused by the new core, no parallel contract implementation |
| WebSocket RPC | Existing `client_server/ws/` from the baseline plus local typed-chunk serialization | One shared protocol/server/client implementation |
| RGB image decode/encode | Existing `utils/process_data.py` | Reused as the only byte codec; no new decoder or channel conversion |
| `src/xpolicy/`, profiles, CLI, tests, validation tool | New local implementation | Core contract/service work; synthetic fixtures explicitly identified |
| EVA | Revision not audited in this change | Migration deferred; no compatibility claim |
| Models / SDKs / weights / URDFs | Policy-specific upstream assets | Not vendored by this change; licenses and runtime acceptance require separate audits |

Git SHA alone does not describe a dirty checkout. The validation tool records hashes of
the actual source files and checks they remain stable during execution. The core lock
and per-environment package inventory are included in the evidence. Installed wheels
are not attributed to the surrounding checkout's git SHA.

## Acceptance scope

| Area | Implemented evidence | Remaining evidence |
| --- | --- | --- |
| A01 public install / CLI | Locked CPU install, lazy imports, CLI help and service startup | Additional supported OS/Python targets |
| A02 XPL RPC | Actual loopback sockets; HELLO, INFER, CALL, RESET, TRIAL_END; reconnect replay, timeout and restart cases | Production network / checkpoint process behavior |
| A03 EVA legacy client | No EVA default client path is introduced | EVA compatibility is deferred |
| A04 dependencies | Separate clean CPU and cloned Conda inventories/reports | Conda model environment retains pre-existing conflicts; G2 SDK environment absent |
| A05 attribution | Original LICENSE/NOTICE retained; tracked source/license/resource inventory and explicit local G2 asset identities, per-scope blockers and pinned license evidence | Original asset/revision provenance, asset-specific terms, scope review and any required permission/review evidence; no blanket redistribution claim |
| B04/B05 boundary validation | Missing cameras, swapped names/order, incompatible robots and unit mismatch rejected | Genuine robot/dataset golden frames |
| C01 full chunk | Counted mock/StarVLA adapter calls; genuine RollingFlow diagnostic returns 16x24 while its planning horizon is 64 | Complete registered G2 runtime constructor/profile and device path |
| C02 API ownership | Separate chunk and legacy services; original legacy result preserved | Original benchmark task baseline |
| C03/C04 protocol faults | Timeout surfaces, same request retries execute once, restart identity rejected | Device-local stale-command behavior |
| Dataset direct / offline scoring | Explicit HDF5 mappings, shared sync/async record-only execution, linked trace and per-field behavior cloning error; synthetic fixture | Genuine calibrated 24D replay profiles and complete runtime adapter |
| Training golden / first divergence | Actual G2 loader samples, raw roundtrips, named statistics, adapter image parity; actual XPL normalization wrapper composed with genuine checkpoint, final VLM/state tensors and float32 output actions compare exactly for three samples | Full adapter constructor/head/waist mapping, physical units, historical training revision and task success |
| C05/C06 record-only execution | Immediate generation invalidation, latest-only pending observations, actual WS reset interleaving, fixed trim/overlap/tick-consumption checks and paced async replay | EVA strategy/physical SDK regression, device-local invalidation and failure behavior |
| Phase 5 task/lifecycle/report foundation | Explicit TaskSpec constraints, native reset before offline READY, idempotent host stop and finalization, blocked/interrupted/unstarted evidence, version 2 hashed reports and raw-sample summaries | Actual multi-scene/environment providers, causal evaluators, scene/robot reset, physical stop ACK, full system metrics and original benchmark regression |
| Environment registry / provider ownership | Lazy implementation/capability metadata, actual recorded timeline provider owning HDF5/record sink/context, provider doctor and traced reset/preparation receipts; separated trial evidence code | Genuine simulator/live feedback contracts and providers; original benchmark/device acceptance |
| Simulator runtime bring-up | Supervised SDK probes, actual SAPIEN CPU steps, cloned Isaac environment with preserved native binary identities and restored vendor package inventory | SAPIEN device creation segfault, Isaac EULA prompt, host libc/GPU requirements, genuine camera/task/provider and original benchmark regression |
| Config/seed experiment suites | Separate sequential Runner processes, explicit plans and snapshots, bounded host process deadlines/stop, nested result binding/verification and grouped summaries | Genuine multi-checkpoint model/benchmark acceptance, causal task/device comparisons and physical stop behavior |
| TinyVLA RGB image boundary | Shared decode before resize in both HDF5 loaders; raw RGB behavior preserved; actual method checks with real Torch and retained before/after shape/color counterexamples | Full loader constructor, augmentation/language backbone, checkpoint training/debug evaluation and original benchmark regression |
| ME-Dex Clean50 image boundary | HDF5 raw arrays and legacy/marked JPEGs decode through the shared RGB helper; regression checks cover fixed bytes, vlen bytes and padded buffers; released checkpoint BGR setting remains explicit | Full distributed training, checkpoint-specific model parity and RoboTwin benchmark regression |
| Robot backend / EVA | Staged source, process, RPC, observation, action, fault, shadow and controlled-motion acceptance plan | No EVA backend, Fake G2, SDK Mock, Shadow or Real provider implemented or accepted |
| Device | Contract schemas and truthful unexecuted report entries | Dataset wire, SDK mappings and on-site safety |

These rows define scope, not blanket PASS for an entire blueprint acceptance item. In
particular, a model-server double does not establish real-checkpoint normalization or
training-loader parity. Mock names and units are not G2 calibration or servo mappings.

The separate G2 diagnostic loads the supplied RollingFlow checkpoint strictly and
uses the actual XPL wrapper methods with the actual G2 training registry. It does not
construct the complete robot adapter. Its selected output contract is float32:
the loader inverse restores float64 with identical numeric values, and the strict
dtype comparison remains FAIL in the report. The old float32 state path's actual
packing/final-input counterexamples are also retained. See [the reproduction guide](training-golden.md).

## Reproduce evidence

Conda development (retain the existing model stack):

```bash
conda activate XPolicyLab
uv pip install --python "$CONDA_PREFIX/bin/python" -e '.[test]'
python tools/validate_core.py --environment conda --output runs/core-validation/conda
```

Clean CPU core, using the public lock:

```bash
uv sync --locked --extra test --default-index https://pypi.org/simple
uv run --no-sync python tools/validate_core.py --environment cpu --output runs/core-validation/cpu
uv run --no-sync python tools/validate_install.py --output runs/core-validation/wheel
```

Both commands write `TEST_REPORT.json`, logs and software inventories. A nonzero dependency
check in the Conda model environment is retained as FAIL. Core contract doctor has a
separate, explicitly limited scope. CI additionally checks WebSocket 14.2; the lock selects
15.0.1. CI definition does not prove a remote CI run succeeded: inspect its uploaded report.

The original demo/StarVLA benchmark path needs an outer `env_cfg/`, benchmark client,
dataset and checkpoint. Those benchmark entry points are not configured in this active
parent workspace. A genuine RoboDojo migration backup is now located and its API is
inspected read-only. Its SDK environment was cloned, and genuine SAPIEN CPU steps were
verified in the existing RoboTwin interpreter. Rendering/task baselines remain unverified:
SAPIEN device creation crashes natively, and Isaac has a pending license prompt plus
host platform/hardware requirements. Core/SDK package conflicts were measured and the
SDK clone restored to its original inventory. See [local runtime evidence](environments.md#local-runtime-evidence).
Other host directories contain checkpoints and datasets. A G2 pair is identified and its actual
training DataLoader is exported by `tools/export_g2_training_golden.py`; see
[Training-source golden samples](training-golden.md) for commands, findings and scope.
`tools/validate_g2_model_boundary.py` extends this evidence to the final model inputs
and checkpoint output through an explicitly identified diagnostic composition.
The pair is not configured as a complete 24D runtime adapter. Real benchmark regression is BLOCKED. Device tests
and on-site acceptance are also BLOCKED. The project goal remains
in progress while these requirements and the later blueprint phases remain outstanding.

Task, lifecycle and report-summary commands are documented in [Tasks and report summaries](experiments.md).
Version 2 reports distinguish execution completion from unmeasured task success;
physical scene/device behavior is not established by offline readiness or host stop.

Asset/license/source inventory commands and their recorded scope are documented in
[Asset evidence inventory](assets.md). BLOCKED declarations and unknown permissions
are retained separately from successful content-hash checks; A05 is not fully accepted.

Environment registry/provider/preflight commands and genuine simulator inspection scope
are documented in [Environment providers](environments.md). Recorded data provider tests
do not establish simulator, EVA Fake G2 or hardware compatibility.

Configuration/seed orchestration, retained process outcomes and closed-suite verification
are documented in [Experiment suites](experiment-suites.md). They reuse the existing Runner
and do not add a simulator, device executor or success judge.
