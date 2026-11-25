#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KNOWACT_ROOT="${KNOWACT_ROOT:-${SCRIPT_DIR}/G05}"

PYTHON_BIN="${KNOWACT_PYTHON:-$(command -v python3)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set KNOWACT_PYTHON to a valid Python executable." >&2
  exit 2
fi

echo "[KnowAct install] python=${PYTHON_BIN}"
echo "[KnowAct install] xpolicylab=${XPL_ROOT}"
echo "[KnowAct install] g05_root=${KNOWACT_ROOT}"

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
if [[ -f "${KNOWACT_ROOT}/pyproject.toml" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${KNOWACT_ROOT}"
fi

cat <<'EOF'

KnowAct adapter-side dependencies are installed.

Before running evaluation, set:

  export KNOWACT_PYTHON=/path/to/python
  export KNOWACT_CKPT_PATH=/path/to/extracted/g05/checkpoint

The vendored G05 checkout under policy/KnowAct/G05 is used by default. Override
KNOWACT_ROOT only if you want to use another compatible G05 checkout.
EOF
