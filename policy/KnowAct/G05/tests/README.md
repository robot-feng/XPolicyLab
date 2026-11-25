# KnowAct / G0.5 Tests and Diagnostics

This directory contains both regression tests and small source-level diagnostic experiments. They are not all benchmark evaluations.

## Regression tests

Run from the XPolicyLab root after installing the XPolicyLab and vendored G05 packages:

```bash
export XPL_ROOT="$PWD"
export PYTHONPATH="$(dirname "$XPL_ROOT"):$XPL_ROOT/policy/KnowAct/G05/src:$XPL_ROOT/policy/KnowAct/G05/tests:${PYTHONPATH:-}"
python -m unittest discover -s policy/KnowAct/G05/tests -p 'test_*.py' -v
```

| File | Coverage |
|---|---|
| `test_native_context.py` | Real G05 model math: causal masks, GDN multi-token cache, MRoPE, full/incremental path, gradients, modality inputs |
| `test_native_policy.py` | `G05Policy`, BAR layout, native ActionCodec, text/action sequencing, rollback, cache isolation |
| `test_v7.py` | Executed-only history, hold-padding, no future target leakage, RoboDojo grouped action padding |
| `test_xpl_integration.py` | XPolicyLab Adapter and deploy loop using simulated IO boundary |
| `test_comparison.py` | Paired manifest, G05 action source, stride matching, paired score validation |

The XPL integration tests use a small processor fixture and simulated environment boundary. They do not exercise the full real robot normalization stack or a live RoboDojo simulator.

## Fixtures

- `native_context_fixture.py`: small Qwen3.5/GDN reference configuration used by source-level tests.
- `native_policy_fixture.py`: actual G05Policy / ActionCodec code with a tiny random configuration and deterministic test text tokenizer. It is not a pretrained model.
- `native_context_imports.py`: optional dependency stubs for CPU source tests.

## Diagnostic experiments (not benchmark scores)

| File | Purpose | Limitation |
|---|---|---|
| `diagnose_v7.py` | Teacher-forcing vs free-running action-token diagnosis | Small random initialization and synthetic turns |
| `experiment_v7.py` | Held-out permutation / same-Episode history diagnostic | Not cross-Episode ICL benchmark |
| `experiment_native_policy.py` | Small real G05Policy action-code training | Not official pretrained G05 performance |
| `codec_audit_v7.py` | Small FSQ / rule-tokenizer dependence and prefix probe | Not a formal pretrained ActionCodec audit |
| `scripts/audit_action_codes.py` | Analyze externally exported paired real action codes | Input must be paired alternatives from the same available context |

Keep raw outputs under `policy/KnowAct/results/` or a new timestamped experiment directory. Never combine synthetic diagnostic results with RoboDojo success metrics.
