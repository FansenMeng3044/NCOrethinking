#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=${MVMOE_CODE_ROOT:-"${script_dir}/Routing-MVMoE"}
source_root=${NCO_SOURCE_ROOT:-"${script_dir}/.."}
python_bin=${MVMOE_PYTHON:-python}
run_base=${MVMOE_RUN_BASE:-/root/autodl-tmp/mvmoe_split_runs}
run_id=${MVMOE_RUN_ID:-formal_fused_4e100_4gpu_$(date -u +%Y%m%d_%H%M%S)}
run_root="${run_base}/${run_id}"
name=mvmoe_4e_split_n100
gpu_set=0,1,2,3
world_size=4

if [[ ! -f "${code_root}/train_split.py" ]]; then
    echo "Missing deployed trainer under ${code_root}" >&2
    exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 4 ]]; then
    echo "This launcher requires four visible GPUs" >&2
    exit 1
fi

mkdir -p "${run_root}/logs" "${run_root}/pids" "${run_root}/status" \
    "${run_root}/outputs/${name}"
printf '%s\n' "${run_root}" > "${run_base}/current_run"
printf '%s\n' "${run_id}" > "${run_root}/status/run_id"
printf '%s\n' running > "${run_root}/status/phase"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/started_at_utc"
git -C "${source_root}" rev-parse HEAD > "${run_root}/status/nco_commit"
git -C "${source_root}" status --porcelain > "${run_root}/status/nco_status_at_start"
git -C "${code_root}" rev-parse HEAD > "${run_root}/status/upstream_commit"
sha256sum \
    "${code_root}/train_split.py" \
    "${code_root}/SplitTrainer.py" \
    "${code_root}/split/constraints.py" \
    "${code_root}/split/decoder.py" \
    "${code_root}/split/triton_backend.py" \
    "${code_root}/split/verifier.py" \
    "${code_root}/split_envs/giant_tour_env.py" \
    "${code_root}/split_models/xy_models.py" \
    > "${run_root}/status/training_code_sha256"

printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${gpu_set}" \
    "distributed=true" \
    "world_size=${world_size}" \
    "model_type=MOE_SPLIT" \
    "problem_size=100" \
    "pomo_size=100" \
    "static_encoder_features=depot_xy,node_xy" \
    "decoder_constraints=B" \
    "split_constraints=B,L,C,TW" \
    "constraint_factorization_version=2" \
    "feasibility_epsilon=0.00001" \
    "split_backend=triton" \
    "split_backend_validation=reference_exact" \
    "epochs=5000" \
    "global_train_episodes=20000" \
    "global_train_batch_size=128" \
    "local_train_batch_size=32" \
    "lr=0.0001" \
    "weight_decay=0.000001" \
    "milestones=4501" \
    "gamma=0.1" \
    "model_save_interval=300" \
    "seed=2023" > "${run_root}/status/${name}.config"

set +e
cd "${code_root}"
export CUDA_VISIBLE_DEVICES="${gpu_set}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"${python_bin}" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${world_size}" \
    --max_restarts=0 \
    train_split.py \
    --ddp \
    --expected_world_size "${world_size}" \
    --problem Train_ALL \
    --model_type MOE_SPLIT \
    --problem_size 100 \
    --pomo_size 100 \
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
    --log_dir "${run_root}/outputs/${name}"
code=$?
printf '%s\n' "${code}" > "${run_root}/status/${name}.exit_code"
printf '%s\n' "${code}" > "${run_root}/status/overall.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/${name}.finished_at_utc"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/status/finished_at_utc"
if [[ ${code} -eq 0 ]]; then
    printf '%s\n' completed > "${run_root}/status/phase"
else
    printf '%s\n' failed > "${run_root}/status/phase"
fi
exit "${code}"
