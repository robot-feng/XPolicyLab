#!/bin/bash
# Add the pinned official RoboMME backend to the existing native Pi_05 venv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/openpi/.venv/bin/python"
RUNTIME="${SCRIPT_DIR}/robomme_runtime"
REVISION=ac846a9a4aec841e483b1dc5d620dfe426bde853
if [[ ! -x "${PYTHON}" ]]; then
    echo "Run (cd policy/Pi_05 && bash install.sh) first" >&2
    exit 1
fi
if [[ ! -d "${RUNTIME}" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/RoboMME/robomme_policy_learning.git "${RUNTIME}"
    git -C "${RUNTIME}" checkout --detach "${REVISION}"
fi
if [[ "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${REVISION}" ]] || [[ -n "$(git -C "${RUNTIME}" status --porcelain)" ]]; then
    echo "Expected clean RoboMME runtime at ${REVISION}; inspect ${RUNTIME}" >&2
    exit 1
fi
uv pip install --python "${PYTHON}" 'hydra-core==1.3.2'
echo "RoboMME backend ready: ${REVISION} (original openpi source unchanged)"
