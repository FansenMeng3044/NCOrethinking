#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=${MVMOE_CODE_ROOT:-"${script_dir}/Routing-MVMoE"}
python_bin=${MVMOE_PYTHON:-python}
run_base=${MVMOE_RUN_BASE:-/root/autodl-tmp/mvmoe_split_runs}
run_id=${MVMOE_RUN_ID:-n100_ddp2_$(date -u +%Y%m%d_%H%M%S)}
run_root="${run_base}/${run_id}"

if [[ ! -f "${code_root}/train_split.py" ]]; then
    echo "Missing deployed trainer under ${code_root}" >&2
    exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 6 ]]; then
    echo "This launcher requires six visible GPUs" >&2
    exit 1
fi

mkdir -p "${run_root}/logs" "${run_root}/pids" "${run_root}/status" "${run_root}/outputs"
mkdir -p "${run_base}"
printf '%s\n' "${run_root}" > "${run_base}/current_run"
printf '%s\n' "${run_id}" > "${run_root}/status/run_id"
printf '%s\n' "running" > "${run_root}/status/phase"
git -C "${code_root}" rev-parse HEAD > "${run_root}/status/upstream_commit"
git -C "${code_root}" status --porcelain > "${run_root}/status/source_status_at_start"

declare -a task_names=()
declare -a task_pids=()

start_task() {
    local name=$1
    local gpu_pair=$2
    local model_type=$3
    local task_output="${run_root}/outputs/${name}"
    mkdir -p "${task_output}"
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES=${gpu_pair}" \
        "distributed=true" \
        "world_size=2" \
        "model_type=${model_type}" \
        "problem_size=100" \
        "pomo_size=100" \
        "static_encoder_features=depot_xy,node_xy" \
        "decoder_constraints=B" \
        "split_constraints=B,L,C,TW" \
        "constraint_factorization_version=2" \
        "epochs=5000" \
        "global_train_episodes=20000" \
        "global_train_batch_size=128" \
        "local_train_batch_size=64" \
        "lr=0.0001" \
        "milestones=4501" \
        "gamma=0.1" \
        "model_save_interval=2500" > "${run_root}/status/${name}.config"

    (
        set +e
        cd "${code_root}" || exit 120
        export CUDA_VISIBLE_DEVICES="${gpu_pair}"
        "${python_bin}" -u -m torch.distributed.run \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=2 \
            --max_restarts=0 \
            train_split.py \
            --ddp \
            --expected_world_size 2 \
            --problem Train_ALL \
            --model_type "${model_type}" \
            --problem_size 100 \
            --pomo_size 100 \
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
            --log_dir "${task_output}"
        code=$?
        printf '%s\n' "${code}" > "${run_root}/status/${name}.exit_code"
        exit "${code}"
    ) > "${run_root}/logs/${name}.log" 2>&1 &

    local pid=$!
    printf '%s\n' "${pid}" > "${run_root}/pids/${name}.pid"
    task_names+=("${name}")
    task_pids+=("${pid}")
}

start_task pomo_mtl_split_n100  "0,1" MTL_SPLIT
start_task mvmoe_4e_split_n100  "2,3" MOE_SPLIT
start_task mvmoe_4el_split_n100 "4,5" MOE_LIGHT_SPLIT

overall=0
for index in "${!task_pids[@]}"; do
    if ! wait "${task_pids[$index]}"; then
        overall=1
    fi
done

if [[ ${overall} -eq 0 ]]; then
    printf '%s\n' "completed" > "${run_root}/status/phase"
else
    printf '%s\n' "failed" > "${run_root}/status/phase"
fi
printf '%s\n' "${overall}" > "${run_root}/status/overall.exit_code"
exit "${overall}"
