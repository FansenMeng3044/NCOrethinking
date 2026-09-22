#!/usr/bin/env bash
set -Eeuo pipefail

E=/root/autodl-tmp/pomo_mtl_split_n100_official_eval_20260922
PY=/root/autodl-tmp/conda_envs/mvmoe/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/root/miniconda3/bin/python
fi

mkdir -p "$E"/{logs,status,results,pids}
date -u +%FT%TZ > "$E/status/started_at_utc"
echo running > "$E/status/phase"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$E/status/exit_code"
  date -u +%FT%TZ > "$E/status/finished_at_utc"
  if [[ "$rc" -eq 0 ]]; then
    echo completed > "$E/status/phase"
  else
    echo failed > "$E/status/phase"
  fi
}
trap finish EXIT

CUDA_VISIBLE_DEVICES=0 bash "$E/run_size.sh" 50 100 > "$E/logs/n50.log" 2>&1 &
p50=$!
printf '%s\n' "$p50" > "$E/pids/n50.wrapper.pid"

CUDA_VISIBLE_DEVICES=1 bash "$E/run_size.sh" 100 32 > "$E/logs/n100.log" 2>&1 &
p100=$!
printf '%s\n' "$p100" > "$E/pids/n100.wrapper.pid"

CUDA_VISIBLE_DEVICES=2 bash "$E/run_size.sh" 200 8 > "$E/logs/n200.log" 2>&1 &
p200=$!
printf '%s\n' "$p200" > "$E/pids/n200.wrapper.pid"

rc=0
wait "$p50" || rc=$?
wait "$p100" || rc=$?
wait "$p200" || rc=$?
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

"$PY" "$E/summarize_eval.py" --results_dir "$E/results"
echo "[$(date -u +%FT%TZ)] all evaluations and summaries completed"
