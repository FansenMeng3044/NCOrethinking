# CVRP size-generalization evaluation

This frozen package evaluates the repository's four final `n=100` checkpoints
(`AM`, `AM-Split`, `POMO`, and `POMO-Split`) on 1,000 deterministic random CVRP
instances at each of `n=200`, `n=500`, and `n=1000`. It contains the generator,
evaluator, orchestration and summarization code, the generated datasets, every
decoded route, logs, aggregate results, and the independent final audit.

## Protocol

- Coordinates are sampled independently and uniformly from the unit square.
- Integer customer demands are sampled uniformly from `{1,...,9}`.
- Vehicle capacity is fixed at 50. This deliberately preserves the training
  distribution's `n=100` demand/capacity semantics while varying only size.
- All evaluations use deterministic greedy decoding and no geometric
  augmentation.
- AM evaluates one trajectory. POMO uses the complete multi-start set,
  `pomo_size=n`, including 1,000 starts for `n=1000`.
- No sampling, beam search, repair, local search, or test-time optimization is
  used.
- Every selected solution is independently replayed using double-precision
  Euclidean distances to verify exact coverage, no duplicate customers,
  capacity feasibility, route count, and reported cost.

All three datasets use seed 1234 and contain 1,000 instances. The checkpoint
and dataset SHA256 values are embedded in every result row and summary.

## Repository checkpoints

| Architecture | Repository path | Epoch | SHA256 |
|---|---|---:|---|
| AM | `artifacts/cvrp_xml100_20260827/checkpoints/am_n100/epoch-100.pt` | 100 | `450c46de4a257605d66b5705cb38f0fd265ec3587a2c7bd866d0605ee463dfce` |
| AM-Split | `artifacts/cvrp_xml100_20260827/checkpoints/am_split_n100/epoch-100.pt` | 100 | `b669a5251c8aba3ba97d74df3c6ea1fc8e0cda0066be1bb76e9855d9d1ce2f44` |
| POMO | `artifacts/cvrp_xml100_20260827/checkpoints/pomo_n100/checkpoint-2000.pt` | 2000 | `74bf36630f179b69d69ceae4997d6b03994815677c0c6bf4000b4936feb29af8` |
| POMO-Split | `artifacts/cvrp_xml100_20260827/checkpoints/pomo_split_n100/checkpoint-2000.pt` | 2000 | `887e1e7a24df1cdb4877b11a188ecda2648e006d8cea7f849866e51332b6fbac` |

## Contents

- `generate_datasets.py`: deterministic dataset generator.
- `evaluate_size_generalization.py`: checkpoint loader, inference driver,
  per-instance atomic writer, and independent route replay.
- `summarize_results.py`: aggregate CSV generator.
- `run_all.sh`: full 12-task evaluation entry point.
- `data/`: the three generated `.pt` datasets plus metadata.
- `results/`: 12 formal tasks, each with 1,000 route JSON files, a result CSV,
  and a summary JSON; `aggregate.csv` is the combined table.
- `smoke/`: four preflight results used to validate all model-loading paths.
- `logs/`: complete evaluator and orchestration logs.
- `status/`: all task exit codes and `final_verification.json`.
- `MANIFEST.sha256`: hashes for the code, datasets, aggregate CSV, and final
  audit report.

The final audit reports 12,000/12,000 unique and feasible model-instance
results, with 1,000 JSON files and 1,000 CSV rows for every task. All 12 task
exit codes and the overall exit code are zero.

## Reproduction

From the repository root, activate a PyTorch environment compatible with the
model code and run:

```bash
python generalization_eval_prep/generate_datasets.py \
  --output-dir generalization_eval_prep/data --seed 1234 --instances 1000
bash generalization_eval_prep/run_all.sh
```

Each instance is written atomically, so rerunning an interrupted task resumes
without recomputing completed rows.
