#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=${MVMOE_CODE_ROOT:-"${script_dir}/Routing-MVMoE"}
source_root=${NCO_SOURCE_ROOT:-"${script_dir}/.."}
python_bin=${MVMOE_PYTHON:-python}
run_base=${MVMOE_RUN_BASE:-/root/autodl-tmp/mvmoe_split_runs}
run_id=${MVMOE_RUN_ID:-formal_fused_mvmoe_4e_n50_1gpu_$(date -u +%Y%m%d_%H%M%S)}
run_root="${run_base}/${run_id}"
task=mvmoe_4e_split_n50
gpu_set=${MVMOE_GPU_SET:-4}

if [[ ! -f "${code_root}/train_split.py" ]]; then
    echo "Missing deployed trainer under ${code_root}" >&2
    exit 1
fi

mkdir -p "${run_root}/logs" "${run_root}/pids" "${run_root}/status" "${run_root}/outputs/${task}"
printf '%s\n' "${run_root}" > "${run_root}/status/run_root"
printf '%s\n' "${run_id}" > "${run_root}/status/run_id"
printf '%s\n' "running" > "${run_root}/status/phase"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/started_at_utc"
git -C "${source_root}" rev-parse HEAD > "${run_root}/status/nco_commit"
git -C "${source_root}" status --porcelain > "${run_root}/status/nco_status_at_start"
git -C "${code_root}" rev-parse HEAD > "${run_root}/status/upstream_commit"
sha256sum \
    "${code_root}/train_split.py" \
    "${code_root}/SplitTrainer.py" \
    "${code_root}/split/decoder.py" \
    "${code_root}/split/triton_backend.py" \
    "${code_root}/split_models/xy_models.py" \
    > "${run_root}/status/training_code_sha256"

printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${gpu_set}" \
    "distributed=false" \
    "world_size=1" \
    "model_type=MOE_SPLIT" \
    "problem_size=50" \
    "pomo_size=50" \
    "static_encoder_features=depot_xy,node_xy" \
    "decoder_constraints=B" \
    "split_constraints=B,L,C,TW" \
    "constraint_factorization_version=2" \
    "feasibility_epsilon=0.00001" \
    "split_backend=triton" \
    "split_backend_validation=bitwise_reference_exact" \
    "epochs=5000" \
    "global_train_episodes=20000" \
    "global_train_batch_size=128" \
    "local_train_batch_size=128" \
    "lr=0.0001" \
    "weight_decay=0.000001" \
    "milestones=4501" \
    "gamma=0.1" \
    "model_save_interval=300" \
    "seed=2023" > "${run_root}/status/${task}.config"

set +e
(
    cd "${code_root}" || exit 120
    export CUDA_VISIBLE_DEVICES="${gpu_set}"
    export PYTHONUNBUFFERED=1
    "${python_bin}" -u train_split.py \
        --problem Train_ALL \
        --model_type MOE_SPLIT \
        --problem_size 50 \
        --pomo_size 50 \
        --epochs 5000 \
        --train_episodes 20000 \
        --train_batch_size 128 \
        --lr 0.0001 \
        --weight_decay 0.000001 \
        --milestones 4501 \
        --gamma 0.1 \
        --model_save_interval 300 \
        --metrics_log_interval 1 \
        --metrics_flush_interval 50 \
        --split_backend triton \
        --seed 2023 \
        --log_dir "${run_root}/outputs/${task}"
) > "${run_root}/logs/${task}.log" 2>&1 &
train_pid=$!
printf '%s\n' "${train_pid}" > "${run_root}/pids/${task}.train.pid"
wait "${train_pid}"
code=$?
printf '%s\n' "${code}" > "${run_root}/status/${task}.exit_code"
printf '%s\n' "${code}" > "${run_root}/status/overall.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/finished_at_utc"
if [[ ${code} -eq 0 ]]; then
    printf '%s\n' "completed" > "${run_root}/status/phase"
else
    printf '%s\n' "failed" > "${run_root}/status/phase"
fi
exit "${code}"
