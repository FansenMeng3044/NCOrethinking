# MVMoE/4E-Split n=100 reproducibility record

This directory archives the final B-mask v5 training run and its same-size
official MVMoE evaluation. The policy uses the original five-dimensional
static customer features and the original dynamic decoder context. Its
construction mask contains only depot/visited validity and the official
backhaul-order feasibility rule. Capacity, time windows, open routes, and
route-duration limits are handled by Split after construction of the customer
order.

## Training

- Model: `MOE_SPLIT` (MVMoE/4E-Split)
- Training variants: CVRP, OVRP, VRPB, VRPL, VRPTW, and OVRPTW
- Problem size and POMO size: 100
- Epochs: 5,000
- Global training episodes per epoch: 20,000
- Global/per-rank batch size: 256/64
- DDP world size: 4
- Seed: 2023
- Checkpoint interval: 300 epochs, plus the final epoch
- Final checkpoint: `training/epoch-5000.pt`
- Training metrics: `training/training_metrics.csv.gz` (lossless gzip)

The run completed 100,000,000 training samples in 218,058.50 seconds (60.57
hours). The metrics archive contains 405,019 rows, reaches epoch 5,000, and has
no non-finite values in applicable optimization fields. Recorded learning
rates are `1e-4` and `1e-5` under the configured schedule.

## Same-size official evaluation

The final size-100 checkpoint is evaluated on the official fixed n=100 data for
all 16 MVMoE variants, with 1,000 instances per variant. Direct and Split use
greedy POMO decoding, `pomo_size=100`, and 8-fold geometric augmentation.

Direct solved 16,000/16,000 instances. Split solved 12,089/16,000, including
all 6,000 instances from the six variants represented during training. The
3,911 unsuccessful cases are retained as model outcomes; they are not repaired
or removed. On the paired instances, Split is 5.453% more costly than Direct
overall, 5.159% on the six training variants, and 5.791% on the ten unseen
variants.

Cross-size results generated before the evaluation scope was narrowed are
preserved on the evaluation host but intentionally excluded from this record
and its aggregate tables.

## Validation

- Training and evaluation exit codes: 0
- Final checkpoint epoch: 5,000
- Checkpoint SHA-256: `b9dbbdb7dc0a5f7a64c484f02794bb45c0e017d77c165798353c6b0d49e8811d`
- `torch.load(..., map_location="cpu", weights_only=False)` succeeded
- Per-instance CSVs each contain 16,000 unique problem-instance rows
- All successful costs and vehicle counts are finite
- Aggregate tables contain 3 regime rows and 16 environment rows

## Files

- `training/epoch-5000.pt`: final checkpoint (Git LFS)
- `training/epoch-5000.pt.resume.json`: strict-resume metadata
- `training/training_metrics.csv.gz`: lossless full training CSV (Git LFS)
- `training/logs/`, `training/status/`, `training/run.sh`: training evidence
- `evaluation/results/`: same-size per-instance and aggregate results
- `evaluation/logs/`, `evaluation/status/`: evaluation logs and status
- `evaluation/scripts/`: exact evaluator, summarizer, and validator
- `MANIFEST.sha256`: hashes for archived source files

