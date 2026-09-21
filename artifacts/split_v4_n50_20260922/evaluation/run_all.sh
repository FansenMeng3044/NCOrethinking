#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/split_v4_n50_eval_20260922
mkdir -p "$ROOT/logs" "$ROOT/pids" "$ROOT/status"
printf 'running\n' > "$ROOT/status/phase"
(
  set +e
  CUDA_VISIBLE_DEVICES=0 bash "$ROOT/run_lane_cvrp_xml.sh" > "$ROOT/logs/lane_cvrp_xml.log" 2>&1
  code=$?
  printf '%s\n' "$code" > "$ROOT/status/lane_cvrp_xml.exit_code"
  exit "$code"
) &
p0=$!
printf '%s\n' "$p0" > "$ROOT/pids/lane_cvrp_xml.wrapper.pid"
(
  set +e
  CUDA_VISIBLE_DEVICES=1 bash "$ROOT/run_lane_tw_solomon.sh" > "$ROOT/logs/lane_tw_solomon.log" 2>&1
  code=$?
  printf '%s\n' "$code" > "$ROOT/status/lane_tw_solomon.exit_code"
  exit "$code"
) &
p1=$!
printf '%s\n' "$p1" > "$ROOT/pids/lane_tw_solomon.wrapper.pid"
set +e
wait "$p0"; c0=$?
wait "$p1"; c1=$?
set -e
if [[ "$c0" -eq 0 && "$c1" -eq 0 ]]; then
  printf 'completed\n' > "$ROOT/status/phase"
  exit 0
fi
printf 'failed cvrp_xml=%s tw_solomon=%s\n' "$c0" "$c1" > "$ROOT/status/phase"
exit 1
