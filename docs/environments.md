# Environment providers

The environment registry exposes implementation status before a runtime is imported.
The Runner currently consumes the `recorded_timeline_v1` contract through the actual
`dataset_direct` provider. This establishes a provider boundary for recorded replay;
it does not establish simulator/device providers or complete Phase 5 acceptance.

```bash
xpolicy environments list
xpolicy environments list --json
```

| Kind | Current status | Runtime evidence |
| --- | --- | --- |
| `dataset_direct` | IMPLEMENTED | Verified HDF5 source, shared RGB decoder, sync/async record-only replay |
| `dataset_wire` | DEFERRED | Genuine EVA/G2 WireObservation/ZMQ migration pending |
| `fake_plant` | DEFERRED | Genuine EVA Fake G2 implementation/regression pending |
| `sdk_mock` | DEFERRED | Original SDK Node injection and mappings pending |
| `physics_sim` | NOT_IMPLEMENTED | Genuine SAPIEN CPU steps verified separately; provider, rendering/task assets and original benchmark regression pending |
| `shadow` | DEFERRED | Verified mappings and NoActuationDriver pending |
| `real` | DEFERRED | Device guard, explicit motion authorization and on-site acceptance pending |

IMPLEMENTED identifies code, not a guarantee that a particular dataset/configuration
is valid. Deferred/unimplemented entries have no callable factory. The Runner refuses
them before importing a policy; it does not switch to a different environment.
Metadata discovery does not import h5py, Torch, Isaac, ROS or a policy adapter.
The staged plan for selecting a device client, preserving its native environment and
validating a future hardware backend is documented in
[Robot Backend integration](robot-backend-integration.md). It does not change the
provider statuses above.

## Provider preflight

```bash
python tools/create_mock_dataset.py --output runs/environment-example --frames 6
xpolicy doctor --mode environment \
  --config runs/environment-example/experiment.yaml \
  --output runs/environment-example/provider-doctor.json
xpolicy eval --config runs/environment-example/experiment.yaml
```

The default doctor mode remains `contract`: it checks declarations and records the
provider's implementation status without creating it. A PASS in that scope does not
claim a deferred environment exists. `--mode environment` additionally admits the
requested provider, verifies the dataset identity and HDF5 stream layout, and closes
the provider. It does not load a policy, decode final model tensors, run a control
loop or score a task. Unknown runtime/physical requirements remain unexecuted.
The public wheel contains the provider implementation and registry; actual benchmark
engines, assets, checkpoints and configs remain separately supplied.

## Recorded timeline ownership

`RecordedDatasetEnvironment` owns the HDF5 handle, source identity checks, recorded
timeline and `RecordOnlySink`. The Runner owns trial states, model resets/calls,
generation allocation and ActionBuffer scheduling. Trial task/evidence code lives in
`evaluation/trial.py`; it no longer accesses an HDF5 decoder or owns another action sink.

At each trial reset, the provider receives an explicit `EpisodeContext` with episode
ID, generation, seed and repeat index. Generations must increase, including when the
episode ID is reused. Reset creates a fresh record-only sink. Recorded reset receipts
have no physical robot/scene reset or device ACK. Stop is idempotent and rejects
subsequent observations/submissions until another valid reset. Close is idempotent.

`time_at(control_tick)` returns the actual recorded frame ID, timestamp and clock.
Control ticks are separate from frame IDs, including nonconsecutive frame IDs.
The provider validates a submission's contiguous control ticks, active episode and
generation, source frame/clock, finite action layout and sequence before accepting it.
Source observations remain independent of submitted actions. Recording creates owned
read-only values/labels plus per-step frame/timestamp receipts; feedback and task
success remain unmeasured. Existing sync prefixes and async timing/trim/overlap behavior
continue through the shared buffer, as documented in [Chunk execution](execution.md).

Provider preparation descriptors and reset receipts are recorded in the same trace as
model/execution events. Results include `environment_provider`; report summaries check
it against the hashed preparation evidence. Existing version 2 reports without this
additional field remain readable. This does not validate a physical scene or renderer.

## Simulator source inspection

A local migration backup contains genuine RoboDojo `TaskEnv` and eval-client source.
Read-only inspection identifies its `create_eval_env(config, app, ...)` factory and
`get_obs` / `take_action` / `is_episode_end` interface. Its reset restores simulation,
scene, robot and cameras, and it requires an Isaac Lab app before construction.
It is materially different from resetting a policy or opening a recorded timeline.

The original debug client's fixed observations/action-shape checks cannot establish
physical simulation or task success. Neither it nor a new synthetic plant replaces
RoboDojo/EVA acceptance. The runtime probes below keep these acceptance scopes separate.

## Simulator runtime probes

Run the supervisor from an installed core environment; the child needs only its own
SDK and standard library. Every invocation requires a new output directory:

```bash
python tools/probe_simulator_runtime.py \
  --python /path/to/simulator/env/bin/python --mode packages \
  --output runs/simulator-packages --timeout-s 30
python tools/probe_simulator_runtime.py \
  --python /path/to/robotwin/env/bin/python --mode sapien_cpu \
  --output runs/sapien-cpu --timeout-s 60
python tools/probe_simulator_runtime.py \
  --python /path/to/robotwin/env/bin/python --mode sapien_render \
  --output runs/sapien-render --timeout-s 120 --gpu 0
```

`packages` captures versions, module locations, wheel tags, host libc and installed native
constraints; PASS means metadata was collected. It does not import Torch or boot an engine.
`sapien_cpu` imports the actual SDK and steps an empty PhysX CPU scene twice.
`sapien_render` requests the actual RoboTwin ray tracing configuration (`rt`, 32 samples,
depth 8, OIDN) and a 64×64 camera in a small probe scene. Its optional NPZ stores the raw
SDK RGBA buffer, not encoded dataset images or a verified public RGB observation.
Neither scene is a benchmark task. `isaac_app` is a separate headless application boot
probe with two app updates and close; it does not construct a RoboDojo task. Vendor
license acceptance is inherited from an explicitly authorized environment, never set by
the tool. An SDK license prompt with closed stdin remains a reported failure/blocker.

The parent retains a frozen child script, stage receipts, log, actual interpreter identity,
tool/shared-supervisor source hashes, exit code/signal and file hashes in `SIMULATOR_PROBE_REPORT.json`.
Native failures retain a Python stack via `faulthandler`. Timeouts terminate only the
owned process group, escalating TERM to KILL; an unreaped root is recorded explicitly.
An SDK worker remaining after root exit also receives termination. Zero exit without
successful child evidence fails. Missing interpreters/packages are BLOCKED; changing
probe source during a run fails source stability. No report overwrites an earlier run.

### Local runtime evidence

The following are host-local observations from 2026-10-14; ignored `runs/` files are not
included in Git and are not portable acceptance results:

| Evidence | Observed result | Scope / remaining requirement |
| --- | --- | --- |
| `runs/simulator-probes/robodojo-clone/CLONE_REPORT.json` | Clone completed after an offline cache miss and exact Conda-package recovery; 29 Conda builds match | 52 protected native versions and selected Torch/CUDA binary hashes match; initial core install creates SDK dependency conflicts |
| `runs/simulator-probes/robodojo-clone/RESTORE_REPORT.json` | SDK package inventory restored exactly | Core removed from SDK interpreter; Isaac Lab editable source still references the migration backup; whole-environment platform checks still FAIL |
| `runs/simulator-probes/robotwin-cpu-02/SIMULATOR_PROBE_REPORT.json` | PASS with stable source | Genuine SAPIEN 3.0.0b1 / PhysX CPU steps only; no task, camera or policy |
| `runs/simulator-probes/robotwin-render-03/SIMULATOR_PROBE_REPORT.json` | FAIL, signal SIGSEGV with stable source | Last recorded stage starts `sapien.Device("cuda:0")`; no device/render-system/frame receipt. The crash cause is not established |
| `runs/simulator-probes/isaac-original-01/runtime.log` | SDK EULA prompt failed on EOF | No app boot. The companion report also fails source stability, so it is not valid successful startup evidence |

The cloned interpreter is `/data/miniconda3/envs/XPolicyLab-RoboDojo/bin/python`
(Python 3.11.15, Torch 2.7.0+cu128). The existing RoboTwin interpreter is
`/data/miniconda3/envs/RoboTwin/bin/python` (Python 3.10.20, SAPIEN 3.0.0b1).
Package installation and recovery targeted only the clone. Migration source was
inspected read-only; SDK probes may write caches/logs at their own default locations.

Isaac wheels declare `manylinux_2_35_x86_64`; this host reports glibc 2.31.
The installed kernel also pins WebSocket 12, incompatible with the public core's
14–15 requirement. Four A100-SXM4-80GB GPUs and driver 580.82.09 were observed.
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
exclude A100/H100 GPUs without RT Cores. CUDA availability and successful CPU physics
therefore do not establish supported Isaac rendering. This documented hardware limit
does not by itself prove the cause of the separate SAPIEN native crash.

Full simulation acceptance still requires a compatible host/runtime, materialized task
assets, genuine scene reset and measured simulation time/feedback, task termination and
native scoring, then an original benchmark comparison under matching seeds and policy
inputs. The `recorded_timeline_v1` provider and `record_only` mode do not supply these
causal semantics; a future simulator provider must declare them explicitly before the
registry exposes a callable factory.
