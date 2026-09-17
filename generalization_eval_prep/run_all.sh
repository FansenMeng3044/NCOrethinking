#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
EVAL="$ROOT/generalization_eval_prep/evaluate_size_generalization.py"
DATA="$ROOT/generalization_eval_prep/data"
RESULTS="$ROOT/generalization_eval_prep/results"
LOGS="$ROOT/generalization_eval_prep/logs"
STATUS="$ROOT/generalization_eval_prep/status"
mkdir -p "$RESULTS" "$LOGS" "$STATUS"

declare -A CHECKPOINTS=(
  [pomo]="$ROOT/artifacts/cvrp_xml100_20260827/checkpoints/pomo_n100/checkpoint-2000.pt"
  [pomo_split]="$ROOT/artifacts/cvrp_xml100_20260827/checkpoints/pomo_split_n100/checkpoint-2000.pt"
  [am]="$ROOT/artifacts/cvrp_xml100_20260827/checkpoints/am_n100/epoch-100.pt"
  [am_split]="$ROOT/artifacts/cvrp_xml100_20260827/checkpoints/am_split_n100/epoch-100.pt"
)

batch_size() {
  case "$1:$2" in
    am:200|am_split:200) echo 32 ;;
    am:500|am_split:500) echo 8 ;;
    am:1000|am_split:1000) echo 2 ;;
    pomo:200|pomo_split:200) echo 8 ;;
    pomo:500|pomo_split:500) echo 2 ;;
    *) echo 1 ;;
  esac
}

overall=0
for size in 200 500 1000; do
  dataset="$DATA/cvrp${size}_n1000_seed1234_cap50.pt"
  for architecture in am am_split pomo pomo_split; do
    task="${architecture}_n100_to_n${size}"
    log="$LOGS/${task}.log"
    echo "$(date --iso-8601=seconds) START $task" | tee -a "$LOGS/orchestrator.log"
    "$PYTHON" "$EVAL" \
      --repo "$ROOT" \
      --dataset "$dataset" \
      --checkpoint "${CHECKPOINTS[$architecture]}" \
      --architecture "$architecture" \
      --output "$RESULTS/$task" \
      --device cuda:0 \
      --batch-size "$(batch_size "$architecture" "$size")" \
      --seed 1234 >>"$log" 2>&1
    code=$?
    printf '%s\n' "$code" >"$STATUS/${task}.exit_code"
    echo "$(date --iso-8601=seconds) END $task exit=$code" | tee -a "$LOGS/orchestrator.log"
    if [[ "$code" -ne 0 ]]; then
      overall="$code"
      break 2
    fi
  done
done

if [[ "$overall" -eq 0 ]]; then
  "$PYTHON" "$ROOT/generalization_eval_prep/summarize_results.py" --root "$RESULTS" \
    >>"$LOGS/orchestrator.log" 2>&1
fi
printf '%s\n' "$overall" >"$STATUS/all.exit_code"
exit "$overall"
