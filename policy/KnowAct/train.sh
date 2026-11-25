#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

if [[ "${action_type}" != "joint" ]]; then
  echo "KnowAct train.sh currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWACT_ROOT="${KNOWACT_ROOT:-${SCRIPT_DIR}/G05}"
PYTHON_BIN="${KNOWACT_PYTHON:-$(command -v python3)}"
OUTPUT_ROOT="${KNOWACT_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
export PYTHON_BIN

if [[ ! -d "${KNOWACT_ROOT}" ]]; then
  echo "KNOWACT_ROOT does not exist: ${KNOWACT_ROOT}" >&2
  exit 3
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set KNOWACT_PYTHON to a valid Python executable." >&2
  exit 3
fi

case "${KNOWACT_TRAIN_BENCHMARK:-}" in
  "")
    case "${bench_name}" in
      RoboDojo|robodojo) train_benchmark="robodojo" ;;
      *) train_benchmark="${bench_name}" ;;
    esac
    ;;
  *) train_benchmark="${KNOWACT_TRAIN_BENCHMARK}" ;;
esac

requested_train_mode="${KNOWACT_TRAIN_MODE:-ar_only}"
case "${requested_train_mode}" in
  fm|fm_only) train_mode="fm_only" ;;
  ar|ar_only|native|context|knowact) train_mode="ar_only" ;;
  ar_fm|ar-fm|ar+fm|cotrain|joint) train_mode="ar_fm" ;;
  *)
    echo "Unsupported training mode: ${requested_train_mode}" >&2
    echo "Use native/ar_only, fm_only, or ar_fm (legacy only)." >&2
    exit 2
    ;;
esac

if [[ "${train_benchmark}" != "robodojo" ]]; then
  echo "This released KnowAct adapter documents RoboDojo simulator training only." >&2
  echo "Set KNOWACT_TRAIN_BENCHMARK=robodojo, or use a compatible external G05 checkout via KNOWACT_ROOT." >&2
  exit 2
fi

if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
  echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset path." >&2
  exit 3
fi
for required in meta data videos; do
  if [[ ! -e "${ROBODOJO_LEROBOT_V30_ROOT}/${required}" ]]; then
    echo "Invalid ROBODOJO_LEROBOT_V30_ROOT: missing ${required}/ under ${ROBODOJO_LEROBOT_V30_ROOT}" >&2
    exit 3
  fi
done

if [[ "${gpu_id}" == *","* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${KNOWACT_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-knowact_robodojo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

if [[ "${WANDB_DIRECT:-1}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

cd "${KNOWACT_ROOT}"

schedule_overrides=()
has_schedule_override=0
for arg in "$@"; do
  case "${arg}" in
    model.max_steps=*|model.max_epochs=*) has_schedule_override=1 ;;
  esac
done
if [[ -n "${KNOWACT_MAX_STEPS:-}" ]]; then
  schedule_overrides+=("model.max_steps=${KNOWACT_MAX_STEPS}" "model.max_epochs=null")
  has_schedule_override=1
fi
if [[ -n "${KNOWACT_MAX_EPOCHS:-}" ]]; then
  schedule_overrides+=("model.max_epochs=${KNOWACT_MAX_EPOCHS}" "model.max_steps=null")
  has_schedule_override=1
fi
if [[ "${has_schedule_override}" != "1" ]]; then
  echo "Set KNOWACT_MAX_STEPS/KNOWACT_MAX_EPOCHS or pass model.max_steps=... / model.max_epochs=... ." >&2
  exit 3
fi

if [[ "${train_mode}" != "ar_only" && "${KNOWACT_NATIVE_CONTEXT:-true}" == "true" ]]; then
  echo "Native KnowAct currently supports AR only; use KNOWACT_NATIVE_CONTEXT=false for legacy FM controls." >&2
  exit 2
fi
case "${train_mode}" in
  fm_only)
    action_overrides=(
      "model.model_arch.discrete_action=false"
      "model.model_arch.continuous_action=true"
      "model.model_arch.fm.joint_training=false"
      "model.model_arch.ar.ce_weight=0.0"
    )
    ;;
  ar_only)
    action_overrides=(
      "model.model_arch.discrete_action=true"
      "model.model_arch.continuous_action=false"
      "model.model_arch.fm.joint_training=false"
      "model.model_arch.fm.fm_weight=0.0"
      "model.model_arch.ar.ce_weight=1.0"
    )
    ;;
  ar_fm)
    action_overrides=(
      "model.model_arch.discrete_action=true"
      "model.model_arch.continuous_action=true"
      "model.model_arch.fm.joint_training=true"
      "model.model_arch.fm.fm_weight=1.0"
      "model.model_arch.ar.ce_weight=1.0"
    )
    ;;
  *)
    echo "Unsupported KNOWACT_TRAIN_MODE=${train_mode}" >&2
    exit 2
    ;;
esac

exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${KNOWACT_TASK_CONFIG:-robodojo_arx_x5_joint}" \
  "seed=${seed}" \
  "exp_name=${EXP_NAME}" \
  "logger.mode=${KNOWACT_LOGGER_MODE:-disabled}" \
  "logger.project=${WANDB_PROJECT}" \
  "model.batch_size=${KNOWACT_BATCH_SIZE:-1}" \
  "model.grad_accumulation_steps=${KNOWACT_GRAD_ACCUM:-8}" \
  "model.model_arch.native_context=${KNOWACT_NATIVE_CONTEXT:-true}" \
  "model.model_arch.native_context_turns=${KNOWACT_CONTEXT_TURNS:-4}" \
  "model.model_arch.native_replan_steps=${KNOWACT_REPLAN_STEPS:-16}" \
  "tokenizer.vq_config.block_wise_autoregressive=true" \
  "tokenizer.vq_config.block_size=16" \
  "model.model_arch.ar.block_wise_autoregressive=true" \
  "batch_size_val=1" \
  "${schedule_overrides[@]}" \
  "${action_overrides[@]}" \
  "$@"
