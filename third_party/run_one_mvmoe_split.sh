#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 RUN_ROOT NAME GPU MODEL_TYPE SIZE ATTEMPT [CHECKPOINT]" >&2
    exit 2
fi

run_root=$1
name=$2
gpu=$3
model_type=$4
size=$5
attempt=$6
checkpoint=${7:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=${MVMOE_CODE_ROOT:-"${script_dir}/Routing-MVMoE"}
python_bin=${MVMOE_PYTHON:-python}
key="${name}.${attempt}"
task_output="${run_root}/outputs/${key}"

case "${model_type}" in
    MTL_SPLIT|MOE_SPLIT|MOE_LIGHT_SPLIT) ;;
    *) echo "Unsupported model type: ${model_type}" >&2; exit 2 ;;
esac
case "${size}" in
    50|100) ;;
    *) echo "Unsupported problem size: ${size}" >&2; exit 2 ;;
esac
if [[ ! "${gpu}" =~ ^[0-5]$ ]]; then
    echo "GPU must be one of 0..5" >&2
    exit 2
fi
if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 2
fi

mkdir -p "${run_root}/logs" "${run_root}/pids" "${run_root}/status" "${task_output}"
printf '%s\n' "$$" > "${run_root}/pids/${key}.pid"
printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${gpu}" \
    "model_type=${model_type}" \
    "problem_size=${size}" \
    "pomo_size=${size}" \
    "attempt=${attempt}" \
    "checkpoint=${checkpoint}" > "${run_root}/status/${key}.config"

extra_args=()
if [[ -n "${checkpoint}" ]]; then
    extra_args+=(--checkpoint "${checkpoint}")
fi

set +e
cd "${code_root}" || exit 120
export CUDA_VISIBLE_DEVICES="${gpu}"
"${python_bin}" -u train_split.py \
    --problem Train_ALL \
    --model_type "${model_type}" \
    --problem_size "${size}" \
    --pomo_size "${size}" \
    --epochs 5000 \
    --train_episodes 20000 \
    --train_batch_size 128 \
    --lr 0.0001 \
    --weight_decay 0.000001 \
    --milestones 4501 \
    --gamma 0.1 \
    --model_save_interval 2500 \
    --metrics_log_interval 1 \
    --metrics_flush_interval 50 \
    --seed 2023 \
    --gpu_id 0 \
    --log_dir "${task_output}" \
    "${extra_args[@]}"
code=$?
printf '%s\n' "${code}" > "${run_root}/status/${key}.exit_code"
exit "${code}"
