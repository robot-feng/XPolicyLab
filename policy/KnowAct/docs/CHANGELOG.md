# Changelog

## V7.1 — Experiment-ready cleanup

- Reorganized the documentation into a Chinese experiment handbook.
- Added source map, data semantics, training/inference protocol, benchmark protocol, reproduction guide, result index and experiment record template.
- Cleaned Python bytecode caches from the release package.
- Reworked `compare.py` into readable, typed functions.
- Reformatted the Native Context, Episode sampler, deploy receipt loop and action-code audit script for maintainability, preserving tested behavior.
- Corrected official RoboDojo G05 comparison default to `action_source=fm`; AR remains an explicit option for an AR-trained checkpoint.
- Added a separate G05-MEM action-source option.
- Added explicit G05 and KnowAct execution-interval matching checks.
- Tightened paired CSV validation and made the cluster-bootstrap interval behavior explicit.
- Extended preflight to report Python compatibility, source presence, asset presence and LeRobot directory layout.
- Invalid training mode values now fail explicitly; `native/context/knowact` are accepted aliases for Native AR training.
- Re-ran the CPU source regression suite: 73 tests passed in this environment.

## V7

- Executed-only history semantics for 32-step plan / 16-step dispatch.
- Native Context training over same-Episode LeRobot history.
- Native G05Policy text/action generation with shared InferenceState.
- BAR16 and original ActionCodec integration.
- XPolicyLab per-environment state isolation and execution receipts.
- Original G05 baseline kept untouched.
