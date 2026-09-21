#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/split_v4_n50_eval_20260922
REPO=/root/autodl-tmp/NCOrethinking_directctx_nomask_v4_20260921
PY=/root/miniconda3/bin/python
DATA=$ROOT/data/fixed/cvrptw50_n10000_seed1234.pt
OUT=$ROOT/results
mkdir -p "$OUT/fixed_tw" "$OUT/solomon50" "$ROOT/logs" "$ROOT/status"

run_am() {
  local model=$1 name=$2
  NCORETHINKING_REPO="$REPO" "$PY" "$ROOT/scripts/fixed/evaluate_fixed_tw_am.py" "$DATA" \
    --model "$model" --num-instances 10000 --batch-size 64 \
    --augmentation 8 --strategy greedy --seed 1234 \
    --output "$OUT/fixed_tw/${name}.json" \
    --per-instance-output "$OUT/fixed_tw/${name}.csv" \
    --timing-warmup-batches 1
}

run_pomo() {
  local architecture=$1 model_dir=$2 name=$3
  "$PY" "$ROOT/scripts/fixed/evaluate_fixed_tw_pomo.py" \
    --repo "$REPO" --dataset "$DATA" --checkpoint-dir "$model_dir" \
    --architecture "$architecture" --instances 10000 --batch-size 32 \
    --augmentation 8 --cuda-device 0 \
    --output "$OUT/fixed_tw/${name}.json" \
    --per-instance-output "$OUT/fixed_tw/${name}.csv" \
    --timing-warmup-batches 1
}

run_am "$ROOT/models/tw/am_direct/epoch-100.pt" am_tw_direct_n50
run_am "$ROOT/models/tw/am_split_v4/epoch-100.pt" am_tw_split_v4_n50
run_pomo pomo_tw "$ROOT/models/tw/pomo_direct" pomo_tw_direct_n50
run_pomo pomo_split_tw "$ROOT/models/tw/pomo_split_v4" pomo_tw_split_v4_n50

"$PY" "$ROOT/scripts/solomon/evaluate_solomon.py" \
  --repo "$REPO" --models "$ROOT/models_solomon.json" \
  --data-root "$ROOT/data/solomon/solomon_tw_eval_20260827_034100/data" \
  --output "$OUT/solomon50" \
  --sintef-bks "$ROOT/scripts/solomon/bks_sintef_double_100.csv" \
  --legacy-reference "$ROOT/scripts/solomon/reference_academic_legacy.csv" \
  --device cuda:0 --augmentation 8 --coordinate-scale 100 --seed 1234 \
  --protocol mvmoe

"$PY" "$ROOT/scripts/solomon/verify_solomon_batch.py" \
  --results "$OUT/solomon50/solomon_results.csv" \
  --solutions "$OUT/solomon50/solutions" \
  --data-root "$ROOT/data/solomon/solomon_tw_eval_20260827_034100/data" \
  --report "$OUT/solomon50/verification_report.json"

"$PY" "$ROOT/scripts/solomon/summarize_results.py" \
  --results "$OUT/solomon50/solomon_results.csv" \
  --output "$OUT/solomon50/solomon_summary.csv"

printf '0\n' > "$ROOT/status/lane_tw_solomon.exit_code"
