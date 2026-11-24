#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_env=${8:-base}
policy_server_port=${9:?policy_server_port required}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"
PYTHON_BIN="${G05_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ "${policy_env}" == /* && -x "${policy_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_env}/bin/python"
  elif [[ -x "${policy_env}" ]]; then
    PYTHON_BIN="${policy_env}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

resolve_ckpt_path() {
  PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - \
    "${SCRIPT_DIR}" "${G05_ROOT}" "${bench_name}" "$1" \
    "${env_cfg_type}" "${action_type}" "${seed}" <<'PY'
import os
from pathlib import Path
import sys
from XPolicyLab.utils.checkpoint_resolver import (
    build_run_dir_name, candidate_checkpoint_roots, ckpt_name_is_path,
)
policy_dir, g05_root, bench, ckpt, robot, action, seed = sys.argv[1:]
cfg = dict(bench_name=bench, ckpt_name=ckpt, env_cfg_type=robot, action_type=action, seed=seed)
explicit = os.environ.get("G05_CKPT_PATH")
if explicit:
    cfg["ckpt_path"] = explicit
candidates = candidate_checkpoint_roots(cfg, Path(policy_dir) / "checkpoints", policy_dir=policy_dir)
if not explicit and not ckpt_name_is_path(ckpt):
    run = build_run_dir_name(cfg)
    if run:
        candidates.append(Path(g05_root) / "outputs" / run)
for candidate in candidates:
    if candidate.exists():
        print(candidate.resolve())
        break
else:
    raise FileNotFoundError("No G05 checkpoint found: " + ", ".join(map(str, candidates)))
PY
}

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 adapter currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G0.5 repo not found: ${G05_ROOT}" >&2
  exit 3
fi

ckpt_path="$(resolve_ckpt_path "${ckpt_name}")"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}" "${action_type}")

echo "[SERVER] policy=${policy_name} task=${task_name} host=${policy_server_host} port=${policy_server_port}"
echo "[SERVER] python=${PYTHON_BIN}"
echo "[SERVER] g05_root=${G05_ROOT}"
echo "[SERVER] ckpt_path=${ckpt_path}"
echo "[SERVER] action_dim=${action_dim}"

cd "${G05_ROOT}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${G05_ROOT}/src:${G05_ROOT}:${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      host="${policy_server_host}" \
      port="${policy_server_port}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      action_dim="${action_dim}" \
      ckpt_path="${ckpt_path}" \
      g05_root="${G05_ROOT}"
