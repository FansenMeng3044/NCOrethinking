#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=${MVMOE_CODE_ROOT:-"${script_dir}/Routing-MVMoE"}
source_root=${NCO_SOURCE_ROOT:-"${script_dir}/.."}
python_bin=${MVMOE_PYTHON:-python}
run_base=${MVMOE_RUN_BASE:-/root/autodl-tmp/mvmoe_split_runs}
run_id=${MVMOE_RUN_ID:-formal_4_1_2_$(date -u +%Y%m%d_%H%M%S)}
run_root="${run_base}/${run_id}"

if [[ ! -f "${code_root}/train_split.py" ]]; then
    echo "Missing deployed trainer under ${code_root}" >&2
    exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 7 ]]; then
    echo "This launcher requires seven visible GPUs" >&2
    exit 1
fi

mkdir -p "${run_root}/logs" "${run_root}/pids" "${run_root}/status" "${run_root}/outputs"
printf '%s\n' "${run_root}" > "${run_base}/current_run"
printf '%s\n' "${run_id}" > "${run_root}/status/run_id"
printf '%s\n' "running" > "${run_root}/status/phase"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/started_at_utc"
git -C "${source_root}" rev-parse HEAD > "${run_root}/status/nco_commit"
git -C "${source_root}" status --porcelain > "${run_root}/status/nco_status_at_start"
git -C "${code_root}" rev-parse HEAD > "${run_root}/status/upstream_commit"
sha256sum \
    "${code_root}/train_split.py" \
    "${code_root}/SplitTrainer.py" \
    "${code_root}/split_models/xy_models.py" \
    > "${run_root}/status/training_code_sha256"

declare -a task_names=()
declare -a task_pids=()

write_config() {
    local name=$1
    local gpu_set=$2
    local world_size=$3
    local model_type=$4
    local problem_size=$5
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES=${gpu_set}" \
        "distributed=$([[ ${world_size} -gt 1 ]] && echo true || echo false)" \
        "world_size=${world_size}" \
        "model_type=${model_type}" \
        "problem_size=${problem_size}" \
        "pomo_size=${problem_size}" \
        "static_encoder_features=depot_xy,node_xy" \
        "decoder_constraints=B" \
        "split_constraints=B,L,C,TW" \
        "constraint_factorization_version=2" \
        "epochs=5000" \
        "global_train_episodes=20000" \
        "global_train_batch_size=128" \
        "local_train_batch_size=$((128 / world_size))" \
        "lr=0.0001" \
        "weight_decay=0.000001" \
        "milestones=4501" \
        "gamma=0.1" \
        "model_save_interval=2500" \
        "seed=2023" > "${run_root}/status/${name}.config"
}

start_task() {
    local name=$1
    local gpu_set=$2
    local world_size=$3
    local model_type=$4
    local problem_size=$5
    local output="${run_root}/outputs/${name}"
    mkdir -p "${output}"
    write_config "${name}" "${gpu_set}" "${world_size}" "${model_type}" "${problem_size}"

    (
        set +e
        cd "${code_root}" || exit 120
        export CUDA_VISIBLE_DEVICES="${gpu_set}"
        export PYTHONUNBUFFERED=1
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        if [[ ${world_size} -gt 1 ]]; then
            "${python_bin}" -u -m torch.distributed.run \
                --standalone \
                --nnodes=1 \
                --nproc_per_node="${world_size}" \
                --max_restarts=0 \
                train_split.py \
                --ddp \
                --expected_world_size "${world_size}" \
                --problem Train_ALL \
                --model_type "${model_type}" \
                --problem_size "${problem_size}" \
                --pomo_size "${problem_size}" \
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
                --log_dir "${output}"
        else
            "${python_bin}" -u train_split.py \
                --problem Train_ALL \
                --model_type "${model_type}" \
                --problem_size "${problem_size}" \
                --pomo_size "${problem_size}" \
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
                --log_dir "${output}"
        fi
        code=$?
        printf '%s\n' "${code}" > "${run_root}/status/${name}.exit_code"
        date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/${name}.finished_at_utc"
        exit "${code}"
    ) > "${run_root}/logs/${name}.log" 2>&1 &

    local pid=$!
    printf '%s\n' "${pid}" > "${run_root}/pids/${name}.wrapper.pid"
    task_names+=("${name}")
    task_pids+=("${pid}")
}

start_task mvmoe_4el_split_n100 "0,1,2,3" 4 MOE_LIGHT_SPLIT 100
start_task pomo_mtl_split_n50 "4" 1 MTL_SPLIT 50
start_task mvmoe_4e_split_n50 "5,6" 2 MOE_SPLIT 50

overall=0
for index in "${!task_pids[@]}"; do
    if ! wait "${task_pids[$index]}"; then
        overall=1
    fi
done

date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/finished_at_utc"
printf '%s\n' "${overall}" > "${run_root}/status/overall.exit_code"
if [[ ${overall} -eq 0 ]]; then
    printf '%s\n' "completed" > "${run_root}/status/phase"
else
    printf '%s\n' "failed" > "${run_root}/status/phase"
fi
exit "${overall}"
