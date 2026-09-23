#!/usr/bin/env bash
set -Eeuo pipefail

: "${EVAL_ROOT:?}"
: "${OFFICIAL_BUNDLE:?}"
: "${SPLIT_SOURCE:?}"
: "${PYTHON:?}"
: "${SPLIT_CHECKPOINT:?}"
: "${DIRECT_CHECKPOINT:?}"
: "${SPLIT_MODEL_TYPE:?}"
: "${DIRECT_MODEL_TYPE:?}"
: "${TRAINED_SIZE:?}"
: "${MODEL_LABEL:?}"

MODE=${1:-three}
mkdir -p "$EVAL_ROOT"/{logs,status,results,pids}
date -u +%FT%TZ > "$EVAL_ROOT/status/started_at_utc"
echo running > "$EVAL_ROOT/status/phase"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$EVAL_ROOT/status/exit_code"
  date -u +%FT%TZ > "$EVAL_ROOT/status/finished_at_utc"
  if [[ "$rc" -eq 0 ]]; then
    echo completed > "$EVAL_ROOT/status/phase"
  else
    echo failed > "$EVAL_ROOT/status/phase"
  fi
}
trap finish EXIT

run_direct() {
  local size=$1 batch=$2 gpu=$3
  date -u +%FT%TZ > "$EVAL_ROOT/status/direct_n${size}.started_at_utc"
  echo running > "$EVAL_ROOT/status/direct_n${size}.phase"
  cd "$OFFICIAL_BUNDLE"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$OFFICIAL_BUNDLE" "$PYTHON" \
    "$EVAL_ROOT/test_direct_csv.py" \
    --checkpoint "$DIRECT_CHECKPOINT" \
    --model_type "$DIRECT_MODEL_TYPE" \
    --problem ALL --problem_size "$size" --pomo_size "$size" \
    --episodes 1000 --batch_size "$batch" --augmentation 8 \
    --data_root "$OFFICIAL_BUNDLE/data" \
    --output "$EVAL_ROOT/results/direct_n${size}.csv" \
    --gpu_id 0 --seed 2024
  echo 0 > "$EVAL_ROOT/status/direct_n${size}.exit_code"
  echo completed > "$EVAL_ROOT/status/direct_n${size}.phase"
  date -u +%FT%TZ > "$EVAL_ROOT/status/direct_n${size}.finished_at_utc"
}

run_split() {
  local size=$1 batch=$2 gpu=$3
  date -u +%FT%TZ > "$EVAL_ROOT/status/split_n${size}.started_at_utc"
  echo running > "$EVAL_ROOT/status/split_n${size}.phase"
  cd "$SPLIT_SOURCE"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$SPLIT_SOURCE" "$PYTHON" \
    "$EVAL_ROOT/test_split_bmask.py" \
    --checkpoint "$SPLIT_CHECKPOINT" \
    --model_type "$SPLIT_MODEL_TYPE" \
    --problem ALL --problem_size "$size" --pomo_size "$size" \
    --episodes 1000 --batch_size "$batch" --augmentation 8 \
    --data_root "$OFFICIAL_BUNDLE/data" \
    --output "$EVAL_ROOT/results/split_n${size}.csv" \
    --gpu_id 0 --seed 2024
  echo 0 > "$EVAL_ROOT/status/split_n${size}.exit_code"
  echo completed > "$EVAL_ROOT/status/split_n${size}.phase"
  date -u +%FT%TZ > "$EVAL_ROOT/status/split_n${size}.finished_at_utc"
}

run_paired() {
  local size=$1 batch=$2 gpu=$3
  run_direct "$size" "$batch" "$gpu"
  run_split "$size" "$batch" "$gpu"
}

launch() {
  local name=$1
  shift
  ( "$@" ) > "$EVAL_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$EVAL_ROOT/pids/${name}.pid"
  PIDS+=("$pid")
}

PIDS=()
if [[ "$MODE" == six ]]; then
  launch direct_n50 run_direct 50 100 0
  launch split_n50 run_split 50 100 1
  launch direct_n100 run_direct 100 32 2
  launch split_n100 run_split 100 32 3
  launch direct_n200 run_direct 200 8 4
  launch split_n200 run_split 200 8 5
elif [[ "$MODE" == three ]]; then
  launch paired_n50 run_paired 50 100 0
  launch paired_n100 run_paired 100 32 1
  launch paired_n200 run_paired 200 8 2
else
  echo "mode must be three or six" >&2
  exit 2
fi

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=$?
done
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

"$PYTHON" "$EVAL_ROOT/summarize_eval.py" \
  --results_dir "$EVAL_ROOT/results" \
  --trained_size "$TRAINED_SIZE" \
  --model "$MODEL_LABEL"
echo "[$(date -u +%FT%TZ)] all evaluations completed"
