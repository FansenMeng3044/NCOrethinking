#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/split_v4_n50_eval_20260922
REPO=/root/autodl-tmp/NCOrethinking_directctx_nomask_v4_20260921
PY=/root/miniconda3/bin/python
DATA=$ROOT/data/fixed/cvrp50_n10000_seed1234.pt
OUT=$ROOT/results
mkdir -p "$OUT/fixed_cvrp" "$OUT/xml100" "$ROOT/logs" "$ROOT/status"
if [[ ! -d "$ROOT/data/xml100/XML/instances" ]]; then
  /root/miniconda3/bin/bsdtar -xf "$ROOT/data/xml100/XML.7z" -C "$ROOT/data/xml100"
fi

run_am() {
  local arch=$1 model=$2 name=$3
  "$PY" "$ROOT/scripts/fixed/evaluate_fixed_cvrp_am.py" \
    --repo "$REPO" --checkpoint "$model" --dataset "$DATA" \
    --architecture "$arch" --instances 10000 --batch-size 128 \
    --augmentation 8 --device cuda:0 --seed 1234 \
    --output "$OUT/fixed_cvrp/${name}.json" \
    --per-instance-output "$OUT/fixed_cvrp/${name}.csv"
}

run_pomo() {
  local arch=$1 model=$2 name=$3
  "$PY" "$ROOT/scripts/fixed/evaluate_fixed_cvrp_pomo.py" \
    --repo "$REPO" --checkpoint "$model" --dataset "$DATA" \
    --architecture "$arch" --instances 10000 --batch-size 64 \
    --augmentation 8 --device cuda:0 --seed 1234 \
    --output "$OUT/fixed_cvrp/${name}.json" \
    --per-instance-output "$OUT/fixed_cvrp/${name}.csv"
}

run_am am "$ROOT/models/cvrp/am_direct/epoch-100.pt" am_direct_n50
run_am am_split "$ROOT/models/cvrp/am_split_v4/epoch-100.pt" am_split_v4_n50
run_pomo pomo "$ROOT/models/cvrp/pomo_direct/checkpoint-2000.pt" pomo_direct_n50
run_pomo pomo_split "$ROOT/models/cvrp/pomo_split_v4/checkpoint-2000.pt" pomo_split_v4_n50

"$PY" "$ROOT/scripts/xml100/evaluate_xml100.py" \
  --repo "$REPO" --models "$ROOT/models_xml100.json" \
  --data-root "$ROOT/data/xml100" \
  --dataset-manifest "$ROOT/scripts/xml100/xml100_dataset_manifest.json" \
  --optimal-costs "$ROOT/scripts/xml100/xml100_optimal_costs.csv" \
  --output "$OUT/xml100" --device cuda:0 --augmentation 8 --seed 1234 \
  --batch-size-pomo 16 --batch-size-am 128 --require-checkpoint-hashes

"$PY" "$ROOT/scripts/xml100/verify_xml100_results.py" \
  --results "$OUT/xml100/xml100_results.csv" \
  --models "$ROOT/models_xml100.json" \
  --data-root "$ROOT/data/xml100" \
  --dataset-manifest "$ROOT/scripts/xml100/xml100_dataset_manifest.json" \
  --optimal-costs "$ROOT/scripts/xml100/xml100_optimal_costs.csv" \
  --report "$OUT/xml100/verification_report.json"

"$PY" "$ROOT/scripts/xml100/summarize_xml100.py" \
  --results "$OUT/xml100/xml100_results.csv" \
  --output-dir "$OUT/xml100/summary"

printf '0\n' > "$ROOT/status/lane_cvrp_xml.exit_code"
