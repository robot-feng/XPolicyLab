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
| `physics_sim` | NOT_IMPLEMENTED | Simulator provider, active runtime/assets and original benchmark regression pending |
| `shadow` | DEFERRED | Verified mappings and NoActuationDriver pending |
| `real` | DEFERRED | Device guard, explicit motion authorization and on-site acceptance pending |

IMPLEMENTED identifies code, not a guarantee that a particular dataset/configuration
is valid. Deferred/unimplemented entries have no callable factory. The Runner refuses
them before importing a policy; it does not switch to a different environment.
Metadata discovery does not import h5py, Torch, Isaac, ROS or a policy adapter.

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

The active simulator checkout/interpreter, relocated assets, render startup, compatible
checkpoint and regression baseline are not verified. The original debug client's
fixed observations/action-shape checks cannot establish physical simulation or task
success. Neither it nor a new synthetic plant replaces RoboDojo/EVA acceptance.
