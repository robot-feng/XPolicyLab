#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

# A named adapter takes precedence over the enclosing benchmark workspace.
case "${bench_name}" in
    ""|.|..|*/*) echo "[ERROR] Expected a benchmark name, got '${bench_name}'" >&2; exit 1 ;;
esac
bench_dir="${root_dir}/XPolicyLab/benchmarks/${bench_name,,}"
entry="${root_dir}/scripts/eval_policy.sh"
if [[ -e "${bench_dir}" || -L "${bench_dir}" ]]; then
    entry="${bench_dir}/scripts/eval_policy.sh"
    export PYTHONPATH="${root_dir}/XPolicyLab:${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ ! -f "${entry}" ]]; then
    echo "[ERROR] Cannot launch '${bench_name}': missing ${entry} (named adapter: ${bench_dir})" >&2
    exit 1
fi

if [[ -f "${eval_env_conda_env}/pyvenv.cfg" ]]; then
    source "${eval_env_conda_env}/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate || true
    conda activate "${eval_env_conda_env}"
fi

echo "[CLIENT] ${entry} (environment: ${eval_env_conda_env})"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

exec bash "${entry}" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --protocol "${protocol}" \
    --eval_batch "${eval_batch}" \
    --root_dir "${root_dir}" \
    --device_id "${env_gpu_id}" \
    --additional_info "${additional_info}" \
    --seed "${seed}"
