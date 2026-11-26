#!/usr/bin/env bash
# Convert native RoboMME HDF5 into episode windows shared with the G05 Panda run.
set -euo pipefail
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == aligned ]]; then
  shift
  export PYTHONPATH="$dir/../..:$dir/../../..${PYTHONPATH:+:$PYTHONPATH}"
  exec "${ROBOMME_PYTHON:?Set ROBOMME_PYTHON to the simulator Python}" \
    "$dir/../../docs/experiments/knowact-robomme/prepare_aligned.py" "$@"
fi
exec "${KNOWACT_PYTHON:-python3}" "$dir/../../docs/experiments/g05-robomme/prepare.py" "$@"
