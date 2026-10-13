# Record-only chunk execution

Both dataset replay strategies use `ActionBuffer` as the single chunk consumer.
Policies return complete `[T,D]` predictions; the execution layer selects, aligns,
optionally blends and consumes them. The sink records scheduled values without an
actuator. This code does not implement EVA's strategy, a hardware command protocol,
GDK mapping, driver watchdog or motion admission.

## Configuration and runnable examples

```yaml
execution:
  mode: record_only
  strategy: async_latest_only  # sync is the regression baseline
  control_hz: 30
  execute_horizon: 2
  latency_trim: true
  overlap_blend_steps: 0
  empty_buffer: skip_record
```

The robot, dataset and control rates must agree. `execute_horizon` must not exceed
the manifest's complete prediction horizon; the blend window must not exceed the
execution horizon. Configuration rejects implicit booleans and unknown empty-buffer
behavior. Non-record-only execution remains unavailable.

```bash
python tools/create_mock_dataset.py --output runs/sync-execution --frames 20
xpolicy eval --config runs/sync-execution/experiment.yaml

python tools/create_mock_dataset.py --output runs/async-execution --frames 20 \
  --strategy async_latest_only
xpolicy eval --config runs/async-execution/experiment.yaml
```

These fixtures are synthetic and use the CPU mock. The existing Conda model stack
and uv/pyproject workflow are unchanged. Reports explicitly identify the source kind,
strategy and control timebase. Recorded sensor frames are independent of predictions;
behavior cloning errors are scored against labels at the actual recorded control row.
Task success, physical feedback and safety are unmeasured.

## Fixed scheduling rules

Let `s` be the observation's control tick, `c` the current control tick, `p` the last
consumed tick, `T` the complete prediction horizon and `E` the execution horizon.
`start_control_tick` records `s`; the request event also records the model dispatch
tick and queue delay. Frame IDs can be sparse and are never used as a clock.

1. The first scheduled tick is `f = max(c, p + 1)`. A tick already consumed, including
   an empty tick, cannot receive a late action.
2. With latency trimming, skip `k = max(0, f - s)` prediction rows. Without trimming,
   `k = 0` and the prefix starts at arrival tick `f`.
3. Schedule rows `[k : min(T, k + E)]` at ticks starting with `f`. If `k >= T`, discard
   the prediction as `prediction_horizon_elapsed`.
4. Each consumed tick yields at most one command. Advancing past unused ticks drops
   those entries and logs the gap. The sink rejects duplicate/out-of-order commands.
5. An accepted newer plan replaces the old future. An older tail cannot resume after
   the newer plan's last step.

Optional overlap is supported only for absolute `joint_position`. For the contiguous
first `n` ticks present in both plans (`n <= overlap_blend_steps`), use
`alpha_i = (i + 1) / (n + 1)` and `output_i = (1 - alpha_i) * old_i + alpha_i * new_i`.
Expired old entries are excluded. Each output retains all contributing chunk IDs,
prediction row indices and cumulative weights. This operation uses float64 arithmetic;
unblended values retain their prediction dtype. Delta, velocity and EEF/quaternion
spaces reject overlap rather than applying this position algorithm to other semantics.

A declared `valid_until_s` is checked at acceptance and consumption. Its clock domain
must match the supplied current time; unavailable/different clocks are rejected.
Trimming uses local control ticks, with no subtraction of timestamps from different
hosts or source clock domains. Missing TTL remains missing: these record-only samples
do not become safe hardware commands.

## Asynchronous lifecycle

`AsyncLatestOnly` keeps one inference call in flight and one pending observation.
Additional observations replace that pending observation, with explicit supersession
events. It validates the same prediction manifest/context as the policy runtime.

Reset increments the local generation and clears actions/queued observations immediately.
The serial worker completes its current call, discards the old-generation result or
error, then performs the newest requested model reset. New-generation inference starts
after that reset. Reset and prediction never run concurrently against the model. This
also works when resetting to the same episode name. A stop/fault clears the buffer;
an explicit new reset is required before scheduling resumes.

The local backend runs the existing synchronous runtime on one dedicated thread.
`WebSocketPolicyBackend` reuses the existing `PolicyEvalClient` CALL/RESET and its request
deduplication/retry/restart rules. Actual loopback tests delay a response across reset
and verify that no old-generation command is consumed. The execution request ID is
also the actual WS frame request ID, verified against the server's response cache.

## Timing and failure evidence

Sync replay uses logical dataset-row control ticks and preserves the old prefix baseline.
It does not interpret synchronous model wall time as elapsed dataset rows. Async replay
uses a paced local monotonic clock: prediction arrival refreshes the control tick even
while sensor sampling waits. If scheduling or decoding misses a tick, recorded frames
are skipped and logged, rather than submitting a burst to catch up. An empty buffer
records a gap; it does not invent held state, zero actions or feedback.

Trace includes A0 complete prediction, A1 trim selection, A2 buffered/blended values,
and A3 ordered record-only commands. A3 is not an SDK-mapped servo request. All stages
retain episode/generation/frame/request/chunk IDs; commands also carry `action_seq` and
contributors. Raw input hashes, configurations, profiles, package inventory and source
hashes remain in the run artifacts.

`results.json` records prediction request counts separately from returned A0 predictions,
actual action counts, empty ticks, skipped frames, superseded
observations and whether inference cleanup drained. Model construction/reset/inference
and cleanup failures retain a FAULT result. Later errors do not overwrite the first
trial fault. NumPy's caller RNG state is restored. Scaled metric reductions avoid
float64 square/sum overflow; metrics outside the JSON float range are null with an
explicit status, not a fabricated score or a policy protocol fault.

An inference timeout stops local consumption without terminating a running vendor
call. If it cannot drain within the trial budget, remaining repeats are not started;
the requested and completed trial counts expose that difference. Policy close is
serialized after inference. A permanently hung local call can retain its worker thread
and delay process exit; process isolation would be required for forced termination.
These checks prove record-only scheduling behavior, not device-local fault handling.
