# Results directory

This directory contains raw V5–V7 diagnostics and historical dry-run artifacts. See [`../docs/09_结果文件索引.md`](../docs/09_结果文件索引.md) for the interpretation and limitations of each file.

- Do not overwrite historical logs.
- Do not mix aborted runs into final statistics.
- Synthetic / small-random-weight diagnostics are not benchmark scores.
- Formal experiment outputs should be stored in a new timestamped directory and include resolved config, source revision, asset hashes, train logs, checkpoint path, eval manifest, paired episode CSV and a written conclusion.

Templates are in `results/templates/`.

The benchmark manifest and dry-run log are portable templates with placeholder paths; run `compare.py` to create machine-specific absolute paths.

Current cleanup regression: `final_regression_v7_1.log` (73 tests passed in the CPU development container). Earlier 54-test and 69-test logs are retained as version history.
