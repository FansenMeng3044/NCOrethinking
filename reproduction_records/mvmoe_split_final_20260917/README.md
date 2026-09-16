# MVMoE-Split final artifacts (2026-09-17)

This directory is the complete reproducibility record for the six final Split
models trained from the MVMoE codebase:

- POMO-MTL-Split, `n=50` and `n=100`;
- MVMoE/4E-Split, `n=50` and `n=100`;
- MVMoE/4E-L-Split, `n=50` and `n=100`.

Every model directory contains the final `epoch-5000.pt`, the complete
`training_metrics.csv.zst`, the training log, and the available status/PID/resume
metadata.  The CSV archives are lossless Zstandard-compressed copies of the
original files.  Large checkpoints and compressed training CSV files are stored
with Git LFS.  Integrity hashes and original byte counts are recorded in
`summary/training_metrics_archives.csv`.

Restore a CSV with:

```bash
zstd -d training_metrics.csv.zst -o training_metrics.csv
```

## Official evaluation

All six final checkpoints were evaluated on the official MVMoE generated
instances for the six training environments and ten zero-shot environments.
Each result CSV contains 16,000 unique `(problem, instance)` rows: 1,000 per
environment.  The common inference protocol is greedy rollout, POMO size equal
to the instance size, 8-fold augmentation, and seed 2024.

The six canonical CSV files are in `evaluations/official_16env/results/`.
Unabridged logs, statuses, smoke tests, evaluation metadata, and the copied
official data archive are retained under `evaluations/raw_by_server/`.

The generated report and machine-readable summaries are in `summary/`:

- `FINAL_REPORT_ZH.md`;
- `official_16env_per_environment.csv`;
- `official_16env_aggregate.csv`;
- `training_and_checkpoint_summary.csv`;
- `published_direct_reference.csv`.

Run the validator/summarizer from the repository root with:

```bash
python reproduction_tools/summarize_mvmoe_split_final.py
```

The script streams all six training CSV archives through `zstd`, validates
result cardinality and finite successful metrics, checks epoch completion, and
recomputes final checkpoint SHA-256 hashes.  It also accepts already restored
`training_metrics.csv` files.

## Source and provenance

`source_snapshots/` preserves the exact server-side code exported with each
run.  `third_party/Routing-MVMoE-split-final/` is a convenient canonical copy
of the final evaluator/trainer implementation.  `provenance/` stores upstream
commits, dirty-tree reports, and tracked patches captured on the three servers.

The upstream MVMoE paper is Zhou et al., *MVMoE: Multi-Task Vehicle Routing
Solver with Mixture-of-Experts*, ICML 2024:
<https://proceedings.mlr.press/v235/zhou24c.html>.

## Important interpretation

`no_feasible_candidate` is a model/evaluation outcome, not a crashed evaluator.
It is concentrated in the four zero-shot environments combining backhauls and
hard time windows.  Conditional means from incomplete environments must not be
reported as full-set means; the generated summary marks this distinction.
