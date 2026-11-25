#!/usr/bin/env bash
# Run from XPolicyLab root; use a new output path for each independent evaluation.
set -euo pipefail
checkpoint=$(realpath -e "${1:?checkpoint file}")
out=$(realpath -m "${2:?new output directory}")
mkdir "$out"
py=${PY:-"$PWD/policy/G05/G05/.venv/bin/python"}
port=$(bash utils/get_free_port.sh)
CUDA_VISIBLE_DEVICES=${POLICY_GPU:-2} PYTHONPATH="$PWD:$PWD/.." "$py" -u -c \
 'import random, runpy, numpy, torch; random.seed(7); numpy.random.seed(7); torch.manual_seed(7); runpy.run_path("setup_policy_server.py",run_name="__main__")' \
 --config_path policy/G05/deploy.yml --port "$port" --overrides "ckpt_path=$checkpoint" eval_embodiment=panda \
 bench_name=RoboMME task_name=MoveCube env_cfg_type=panda action_type=joint action_source=fm frequency=20 seed=7 >"$out/server.log" 2>&1 &
pid=$!; trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
bash utils/wait_for_policy_server.sh localhost "$port" "$pid" G05 900
export EVAL_ENV_TYPE=sim ROBOMME_SPLIT="${ROBOMME_SPLIT:-test}" ROBOMME_EPISODES=1 ROBOMME_ACTION_STEPS=16 ROBOMME_MAX_STEPS=1300
export ROBOMME_EVALUATION_ID="$(basename "$out")"
read -r -a tasks <<< "${ROBOMME_TASKS:-MoveCube PatternLock RouteStick InsertPeg}"
read -r -a episodes <<< "${ROBOMME_EPISODE_IDS:-0 2 3}"
for task in "${tasks[@]}"; do
  for ep in "${episodes[@]}"; do
    ROBOMME_START_EPISODE=$ep ROBOMME_OUTPUT_DIR="$out/$task-$ep" \
      bash policy/G05/setup_eval_env_client.sh RoboMME "$task" external panda joint 7 "${ENV_GPU:-3}" \
      "${SIM_ENV:-$PWD/../robomme_benchmark/.venv}" ckpt_name=external,action_type=joint "$port" localhost >"$out/$task-$ep.log" 2>&1
  done
done
