# Experiment suites

`ExperimentSuiteRunner` schedules ordinary `ExperimentRunner` evaluations sequentially,
using one child process per configuration/base seed. It provides multi-config/seed
orchestration and retained evidence. It does not add another action executor, environment
provider or task judge. Current runnable evaluation remains dataset direct / record only;
unimplemented simulator/device providers retain their normal admission failures.

## Runnable example

From the installed core environment, use a new directory:

```bash
python tools/create_mock_dataset.py --output runs/suite-demo --frames 12 --with-suite
xpolicy eval suite --config runs/suite-demo/suite.yaml
xpolicy results verify-suite --report runs/suite-demo/suite-results/SUITE_REPORT.json
```

The fixture explicitly declares a synthetic dataset, mock policy, two execution strategies
and two base seeds. It produces four separate runs. A PASS verifies this orchestration and
retained record-only evidence; task success remains null, and no original benchmark,
checkpoint inference, scene reset, EVA or hardware is established by this example.

## Suite configuration

```yaml
schema_version: 1
name: checkpoint_seed_comparison
artifacts_root: suite-results
process_timeout_s: 300.0
cleanup_grace_s: 2.0
fail_fast: false
cases:
  - case_id: baseline
    config: experiment.yaml
    seeds: [3, 7]
    repeats: 1
  - case_id: candidate
    config: candidate.yaml
    seeds: [3, 7]
    repeats: 1
    process_timeout_s: 600.0
    python: /path/to/compatible/conda/env/bin/python
```

Suite paths resolve relative to the suite YAML. Each experiment's paths resolve relative
to its own source YAML before a generated case configuration is written. Original configs
are not edited. Each case can select its own task, dataset, policy and checkpoint through
a complete experiment config; a checkpoint still needs its matching model manifest,
training/normalization identities and the existing adapter. No arbitrary checkpoint-list
override bypasses those bindings. Different conditions remain separate summary groups.

`python` is optional and defaults to the supervisor's interpreter. A selected Conda/venv
must already contain the public core and the case's dependencies. The interpreter path
and binary identity are pinned, while executable symlinks are preserved so a venv does
not silently become its base environment. Child Runner source bytes must match the suite
plan. Each run records its actual Python, executable, prefix, platform/libc and packages;
different inventories remain separate summary conditions. No packages are installed or
SDK license flags changed by the suite.

Case IDs use 1–64 ASCII letters/digits/underscore/dot/hyphen, starting with a letter or
digit. IDs and each case's seeds must be unique. Seeds are uint32 integers. `repeats`, when
omitted, preserves the source experiment setting. It counts trials within that one run;
it does not expand the seed list. Seed overrides preserve the existing Runner's NumPy
base seeding and hook metadata, without claiming Torch/CUDA or physical determinism.

`process_timeout_s` covers child startup, all repeats, cleanup and process exit. It is
independent of the per-trial effective task timeout. On deadline or host stop, the parent
sends SIGINT to its owned process group, then TERM and KILL as needed, allowing the configured
grace for each step. It records actual exits and whether the root was reaped. An unreaped
root prevents further scheduling even if its partial report is invalid. This supervision
requires POSIX process groups. Process stopping does not establish a device stop or ACK.

With `fail_fast: false`, ordinary failed/blocked cases do not erase or prevent later cases.
With `true`, remaining cases stay NOT_RUN with their reason. Explicit stop requests and
changed input identities stop scheduling in either setting. Missing/invalid per-case
configs remain BLOCKED entries, with unknown repeat counts left null. A process timeout
remains TIMED_OUT even if a child report was written before process exit stalled.

## Evidence and verification

The new suite directory contains:

| Artifact | Recorded scope |
| --- | --- |
| `source-suite.yaml`, `inputs/<case_id>/` | Original suite/experiment/profile snapshots |
| `plan.json` | Explicit case/seed population, historical path resolutions, effective configs, input/core/lock identities and deadlines |
| `cases/<case_id>/seed-<seed>/experiment.yaml` | Validated config with explicit seed/repeat/artifact-root overrides |
| `cases/<case_id>/seed-<seed>/process.log` | Actual child stdout/stderr |
| `cases/<case_id>/seed-<seed>/artifacts/<run_id>/` | Unchanged version 2 Runner result, trace, profiles, images and inventory |
| `SUITE_REPORT.json` | Scheduling/process outcomes, verified child bindings, summary, populations, source stability and file hashes |

The report is checkpointed while active and finalized once. Reusing an existing output
directory is rejected. The closed directory is a retained artifact: `verify-suite` checks
its file set/hashes, source snapshots, plan/case/seed/config/profile bindings, nested
version 2 evidence, process receipts, outcome counts and recomputed summary. Complete
directories can be moved together, including after the original config/data paths are
removed. Verification reconstructs overrides from retained source snapshots and captured
path-resolution receipts; it does not resolve another host's current symlinks or home
directory. It does not reopen historical external checkpoints or independently verify
the operating system's past path-resolution/process events.

Successful exit without a verified run cannot count as COMPLETED. Wrong seed/config/profile
results cannot enter the summary. Invalid child evidence remains a failed case, and
subsequent verification rejects that invalid evidence. Summary trial denominators include
only verified child runs; suite case denominators also retain unlaunched, crashed and
timed-out cases. No missing trial or task score is fabricated. Failure takes precedence
over blocked/stopped/healthy outcomes.

```python
from xpolicy import ExperimentSuiteRunner

runner = ExperimentSuiteRunner("runs/suite-demo/suite.yaml")
report = runner.evaluate()
```

Each instance evaluates once; subsequent calls return its result or re-raise its error.
Concurrent evaluation on that instance is rejected. `runner.stop()` requests host stopping
from another thread. The CLI handles SIGINT and SIGTERM, forwards the stop through the
supervisor and restores prior signal handlers. Abruptly killing the parent with SIGKILL
cannot run its cleanup handler; the report does not claim that path is safely drained.

The suite is an experiment orchestration foundation for the blueprint. Multi-checkpoint
model acceptance, causal simulator comparisons, device safeguards and the full multi-robot
platform remain separate requirements.
