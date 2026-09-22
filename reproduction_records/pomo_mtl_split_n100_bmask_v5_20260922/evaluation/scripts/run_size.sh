#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 PROBLEM_SIZE BATCH_SIZE" >&2
  exit 2
fi

SIZE="$1"
BATCH="$2"
E=/root/autodl-tmp/pomo_mtl_split_n100_official_eval_20260922
B="$E/official_bundle"
R=/root/autodl-tmp/Routing-MVMoE_bmask_v5_20260921_182920
PY=/root/autodl-tmp/conda_envs/mvmoe/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/root/miniconda3/bin/python
fi
SPLIT_CK=/root/autodl-tmp/mvmoe_split_bmask_v5_batch256_20260921/mtl_split_n100/outputs/20260921_185958_mtl_split_n100/epoch-5000.pt
DIRECT_CK="$B/pretrained/pomo_mtl_n100/epoch-5000.pt"

mkdir -p "$E"/{logs,status,results,pids}
date -u +%FT%TZ > "$E/status/n${SIZE}.started_at_utc"
echo running > "$E/status/n${SIZE}.phase"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$E/status/n${SIZE}.exit_code"
  date -u +%FT%TZ > "$E/status/n${SIZE}.finished_at_utc"
  if [[ "$rc" -eq 0 ]]; then
    echo completed > "$E/status/n${SIZE}.phase"
  else
    echo failed > "$E/status/n${SIZE}.phase"
  fi
}
trap finish EXIT

echo "[$(date -u +%FT%TZ)] Direct trained-n100 evaluated at n=$SIZE batch=$BATCH"
cd "$B"
PYTHONPATH="$B" "$PY" "$E/test_direct_csv.py" \
  --checkpoint "$DIRECT_CK" \
  --problem ALL \
  --problem_size "$SIZE" \
  --pomo_size "$SIZE" \
  --episodes 1000 \
  --batch_size "$BATCH" \
  --augmentation 8 \
  --data_root "$B/data" \
  --output "$E/results/direct_n${SIZE}.csv" \
  --gpu_id 0 \
  --seed 2024

echo "[$(date -u +%FT%TZ)] Split trained-n100 evaluated at n=$SIZE batch=$BATCH"
cd "$R"
PYTHONPATH="$R" "$PY" "$E/test_split_v5.py" \
  --checkpoint "$SPLIT_CK" \
  --model_type MTL_SPLIT \
  --problem ALL \
  --problem_size "$SIZE" \
  --pomo_size "$SIZE" \
  --episodes 1000 \
  --batch_size "$BATCH" \
  --augmentation 8 \
  --data_root "$B/data" \
  --output "$E/results/split_n${SIZE}.csv" \
  --gpu_id 0 \
  --seed 2024

echo "[$(date -u +%FT%TZ)] n=$SIZE complete"
