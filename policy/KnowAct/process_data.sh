#!/usr/bin/env bash
# Convert native RoboMME HDF5 into episode windows shared with the G05 Panda run.
set -euo pipefail
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${KNOWACT_PYTHON:-python3}" "$dir/../../docs/experiments/g05-robomme/prepare.py" "$@"
