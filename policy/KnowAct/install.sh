#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KNOWACT_ROOT="${KNOWACT_ROOT:-${SCRIPT_DIR}/G05}"

# Reuse an existing G05 dependency installation while keeping KnowAct first
# on sys.path. An appended site-packages directory does not execute its editable .pth.
if [[ "${1:-}" == --reuse-g05 ]]; then
  base_py=${2:?Pass the existing G05 Python executable}
  fla_dir=${3:?Pass the directory containing fla-core 0.6.0}
  env_dir="${KNOWACT_ENV:-$SCRIPT_DIR/.venv}"
  [[ -x "$env_dir/bin/python" ]] || "$base_py" -m venv --without-pip "$env_dir"
  base_lib=$("$base_py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  own_lib=$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  realpath -e "$KNOWACT_ROOT/src" "$KNOWACT_ROOT" "$XPL_ROOT/.." "$XPL_ROOT" "$fla_dir" "$base_lib" > "$own_lib/knowact.pth"
  "$env_dir/bin/python" - "$KNOWACT_ROOT" <<'PYTHON'
from importlib.metadata import version
from pathlib import Path
import sys
import g05, torch
if not Path(g05.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise RuntimeError("g05 imports from a different checkout")
if version("fla-core") != "0.6.0":
    raise RuntimeError("Expected fla-core 0.6.0")
print("KnowAct", g05.__file__, "torch", torch.__version__, "fla-core", version("fla-core"))
PYTHON
  exit
fi

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
