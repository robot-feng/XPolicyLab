# Migration status and provenance

## Scope

This checkout is the `robot-feng/XPolicyLab` fork. The v4 blueprint is a design and
acceptance specification; the full unified platform is not implemented. The current
change implements the public core and policy-service foundation. EVA changes were
deferred by the user. No EVA robot, transport, SDK, recorder or source archive is
imported, modified or bundled in this change.

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
| A05 attribution | Original LICENSE retained, NOTICE and file provenance documented | External asset and vendor license inventory |
| B04/B05 boundary validation | Missing cameras, swapped names/order, incompatible robots and unit mismatch rejected | Genuine robot/dataset golden frames |
| C01 full chunk | Counted mock and real StarVLA adapter with a model-server double | Genuine checkpoint forward and training-loader tensor comparison |
| C02 API ownership | Separate chunk and legacy services; original legacy result preserved | Original benchmark task baseline |
| C03/C04 protocol faults | Timeout surfaces, same request retries execute once, restart identity rejected | Device-local stale-command behavior |
| Dataset direct / offline scoring | Explicit HDF5 mappings, record-only prefixes, linked trace and per-field behavior cloning error; synthetic fixture | Genuine training-source golden data and checkpoint parity |
| Device / execution | Contract schemas and truthful unexecuted report entries | Dataset wire, generation-aware asynchronous execution, SDK mappings and on-site safety |

These rows define scope, not blanket PASS for an entire blueprint acceptance item. In
particular, a model-server double does not establish real-checkpoint normalization or
training-loader parity. Mock names and units are not G2 calibration or servo mappings.

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
dataset and checkpoint. The outer benchmark client/configuration is absent here; other
host directories contain checkpoints and datasets, but a compatible pair has not yet
been configured for this project. Real benchmark regression is BLOCKED. Device tests
and on-site acceptance are also BLOCKED. The project goal remains
in progress while these requirements and the later blueprint phases remain outstanding.
