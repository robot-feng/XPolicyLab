#!/usr/bin/env bash
# Resume the four-GPU MoveCube run and test the actual stage weights in the simulator.
set -euo pipefail
exp=$(cd "$(dirname "$0")" && pwd)
cd "$exp/../../.."
: "${KNOWACT_INITIAL:?}" "${KNOWACT_OUTPUT_ROOT:?}" "${KNOWACT_ROBOMME_DATA:?}" "${KNOWACT_PYTHON:?}" "${SIM_PY:?}" "${OLD:?Set OLD to the original 30K weights}"
export KNOWACT_MAX_STEPS=30000 KNOWACT_BATCH_SIZE=1
export PYTHONPATH="$exp:$PWD:$PWD/..:${PYTHONPATH:-}"
export KNOWACT_NATIVE_CONTEXT=true KNOWACT_KEEP_HISTORY=true KNOWACT_REPLAN_STEPS=16 KNOWACT_CONTEXT_MAX_TOKENS=65536
unset KNOWACT_DEMO_HEAD
name=knowact-movecube-fullvlm
export EXP_NAME="RoboMME-$name-panda-joint-7"
run="$KNOWACT_OUTPUT_ROOT/$EXP_NAME"
out="$PWD/runs/robomme/knowact/movecube-full-vlm"
mkdir -p "$out"
exec 9>"$out/driver.lock"
flock -n 9 || { echo 'Another driver holds this run'; exit 1; }
evaluate() (
    set -e
    dir="$out/eval-$1"
    mkdir -p "$dir"
    port=$(bash utils/get_free_port.sh)
    KNOWACT_CKPT_PATH="$run/last.pt" bash policy/KnowAct/setup_eval_policy_server.sh RoboMME MoveCube "$name" panda joint 7 0 "$KNOWACT_PYTHON" "$port" localhost > "$dir/server.log" 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
    bash utils/wait_for_policy_server.sh localhost "$port" "$server" KnowAct 300
    for split in train val; do
        ids=(1 4); if [[ "$split" == val ]]; then ids=(1 8); fi
        CUDA_VISIBLE_DEVICES=1 "$SIM_PY" "$exp/probe_grasp.py" --task MoveCube --split "$split" --episodes "${ids[@]}" --port "$port" --out "$dir" --video
    done
)
step=0
if [[ -f "$run/last.pt" ]]; then
    step=$("$KNOWACT_PYTHON" -c 'import torch,sys; print(torch.load(sys.argv[1],mmap=True,weights_only=True,map_location="cpu")["step"])' "$run/last.pt")
fi
for target in 4 128 1000 10000 20000 30000; do
    if (( step < target )); then
        flags=()
        if (( step > 0 )); then flags+=(--resume); fi
        bash policy/KnowAct/train.sh RoboMME "$name" panda joint 7 0,1,2,3 --task MoveCube \
            --full-vlm --hf-image-patches --warm-start "$OLD" --lr 0.00001 --save-every 512 \
            --val-windows 32 --stop-at "$target" "${flags[@]}"
        step=$target
    fi
    if (( step == target && target > 4 )) && [[ ! -f "$out/eval-$target/complete" ]]; then
        evaluate "$target" > "$out/eval-$target.log" 2>&1
        touch "$out/eval-$target/complete"
    fi
done
