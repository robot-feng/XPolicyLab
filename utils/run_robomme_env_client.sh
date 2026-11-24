#!/bin/bash
set -euo pipefail
eval_batch=$1
eval_env=$2
port=$3
task_name=$5
env_cfg_type=$6
policy_name=$7
additional_info=$8
root_dir=$9
seed=${10}
env_gpu_id=${11}
host=${12:-localhost}
protocol=${13:-ws}
if [[ "${protocol}" != ws || "${env_cfg_type}" != panda || ",${additional_info}," != *,action_type=joint,* ]]; then
    echo "RoboMME client requires protocol=ws, env_cfg_type=panda, action_type=joint" >&2
    exit 1
fi
source "${root_dir}/XPolicyLab/utils/activate_eval_env.sh"
activate_eval_env "${eval_env}"
export PYTHONPATH="${root_dir}/XPolicyLab:${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
args=()
IFS=',' read -r -a overrides <<< "${additional_info}"
for entry in "${overrides[@]}"; do
    if [[ "${entry}" == ckpt_name=* ]]; then
        args+=(--checkpoint "${entry#ckpt_name=}")
    fi
done
if [[ -n "${ROBOMME_OUTPUT_DIR:-}" ]]; then
    args+=(--output "${ROBOMME_OUTPUT_DIR}")
fi
exec "${XPL_EVAL_PYTHON}" "${root_dir}/XPolicyLab/utils/robomme_env_client.py" \
    --task "${task_name}" --policy-name "${policy_name}" --seed "${seed}" \
    --host "${host}" --port "${port}" --eval-batch "${eval_batch}" \
    --split "${ROBOMME_SPLIT:-test}" --episodes "${ROBOMME_EPISODES:-1}" \
    --start-episode "${ROBOMME_START_EPISODE:-0}" --max-steps "${ROBOMME_MAX_STEPS:-1300}" \
    --action-steps "${ROBOMME_ACTION_STEPS:-16}" "${args[@]}"
