#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"

if [[ "${1:-}" == "--inference" ]]; then
  command -v uv >/dev/null || { echo "Install uv first." >&2; exit 2; }
  if [[ ! -x "${G05_ROOT}/.venv/bin/python" ]]; then
    uv venv --python 3.10 --seed "${G05_ROOT}/.venv"
  fi
  PYTHON_BIN="${G05_ROOT}/.venv/bin/python"
  uv pip install --python "${PYTHON_BIN}" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "${SCRIPT_DIR}/requirements-inference.txt"
  uv pip install --python "${PYTHON_BIN}" --no-deps -e "${G05_ROOT}" -e "${XPL_ROOT}"
  "${PYTHON_BIN}" -c 'from g05.models.g05.inferencer import PolicyInferencer; import XPolicyLab; print("G05 inference imports OK")'
  echo "[G05] Activate: source ${G05_ROOT}/.venv/bin/activate"
  exit 0
fi

PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 2
fi

echo "[G05 install] python=${PYTHON_BIN}"
echo "[G05 install] xpolicylab=${XPL_ROOT}"
echo "[G05 install] g05_root=${G05_ROOT}"

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
if [[ -f "${G05_ROOT}/pyproject.toml" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${G05_ROOT}"
fi

cat <<'EOF'

G05 adapter-side dependencies are installed.

Before running evaluation, set:

  export G05_PYTHON=/path/to/python
  export G05_CKPT_PATH=/path/to/extracted/g05/checkpoint

The vendored G05 checkout under policy/G05/G05 is used by default. Override
G05_ROOT only if you want to use another compatible G05 checkout.
EOF
