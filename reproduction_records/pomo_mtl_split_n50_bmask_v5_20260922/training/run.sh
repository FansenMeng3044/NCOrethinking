#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/Routing-MVMoE_bmask_v5_20260921_182920
export CUDA_VISIBLE_DEVICES=0
printf 'running\n' > /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/status/phase
printf '%s\n' '/root/miniconda3/bin/python train_split.py --problem Train_ALL --model_type MTL_SPLIT --problem_size 50 --pomo_size 50 --epochs 5000 --train_episodes 20000 --train_batch_size 128 --model_save_interval 300 --split_backend triton --metrics_log_interval 1 --metrics_flush_interval 50 --seed 2023 --log_dir /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/outputs --gpu_id 0' > /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/status/command.txt
set +e
/root/miniconda3/bin/python train_split.py --problem Train_ALL --model_type MTL_SPLIT --problem_size 50 --pomo_size 50 --epochs 5000 --train_episodes 20000 --train_batch_size 128 --model_save_interval 300 --split_backend triton --metrics_log_interval 1 --metrics_flush_interval 50 --seed 2023 --log_dir /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/outputs --gpu_id 0 >> /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/logs/train.log 2>&1
rc=$?
printf '%s\n' "$rc" > /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/status/exit_code
if [ "$rc" -eq 0 ]; then printf 'completed\n' > /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/status/phase; else printf 'failed\n' > /root/autodl-tmp/mvmoe_split_bmask_v5_20260921/mtl_split_n50/status/phase; fi
exit "$rc"
