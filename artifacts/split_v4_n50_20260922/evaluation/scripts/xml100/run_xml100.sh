#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 NCORETHINKING_REPO MODELS_JSON OUTPUT_DIR DEVICE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$1"
MODELS="$2"
OUTPUT="$3"
DEVICE="$4"
EXPECTED_REPO_COMMIT="${EXPECTED_REPO_COMMIT:-ad0abe30c9eb4d2f9b55cd79239b260b8e30d69c}"
EXPECTED_REPO_BRANCH="${EXPECTED_REPO_BRANCH:-codex/cvrptw-multivehicle-env}"

python "$SCRIPT_DIR/prepare_xml100.py" --no-download --no-extract
python "$SCRIPT_DIR/evaluate_xml100.py" \
  --repo "$REPO" \
  --models "$MODELS" \
  --data-root "$SCRIPT_DIR/data" \
  --dataset-manifest "$SCRIPT_DIR/xml100_dataset_manifest.json" \
  --optimal-costs "$SCRIPT_DIR/xml100_optimal_costs.csv" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --augmentation 8 \
  --seed 1234 \
  --expected-repo-commit "$EXPECTED_REPO_COMMIT" \
  --expected-repo-branch "$EXPECTED_REPO_BRANCH" \
  --require-clean-repo \
  --require-checkpoint-hashes
python "$SCRIPT_DIR/verify_xml100_results.py" \
  --results "$OUTPUT/xml100_results.csv" \
  --models "$MODELS" \
  --data-root "$SCRIPT_DIR/data" \
  --dataset-manifest "$SCRIPT_DIR/xml100_dataset_manifest.json" \
  --optimal-costs "$SCRIPT_DIR/xml100_optimal_costs.csv"
python "$SCRIPT_DIR/summarize_xml100.py" \
  --results "$OUTPUT/xml100_results.csv" \
  --output-dir "$OUTPUT"
