# MVMoE/4E-Split n=50 reproducibility record

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
- Problem size and POMO size: 50
- Epochs: 5,000
- Global training episodes per epoch: 20,000
- Global/per-rank batch size: 128/64
- DDP world size: 2
- Seed: 2023
- Checkpoint interval: 300 epochs, plus the final epoch
- Final checkpoint: `training/epoch-5000.pt`
- Training metrics: `training/training_metrics.csv.gz` (lossless gzip)

The run completed 100,000,000 training samples in 213,462.08 seconds (59.30
hours). The metrics archive contains 795,019 rows, reaches epoch 5,000, and has
no non-finite values in applicable optimization fields. Recorded learning
rates are `1e-4` and `1e-5` under the configured schedule.

## Same-size official evaluation

The final size-50 checkpoint is evaluated on the official fixed n=50 data for
all 16 MVMoE variants, with 1,000 instances per variant. Direct and Split use
greedy POMO decoding, `pomo_size=50`, and 8-fold geometric augmentation.

Direct solved 16,000/16,000 instances. Split solved 12,952/16,000, including
all 6,000 instances from the six variants represented during training. The
3,048 unsuccessful cases are retained as model outcomes; they are not repaired
or removed. On the paired instances, Split is 3.663% more costly than Direct
overall, 2.608% on the six training variants, and 4.671% on the ten unseen
variants.

Cross-size results generated before the evaluation scope was narrowed are
preserved on the evaluation host but intentionally excluded from this record
and its aggregate tables.

## Validation

- Training and evaluation exit codes: 0
- Final checkpoint epoch: 5,000
- Checkpoint SHA-256: `a07a269551811d3fb4531c4106971fd8ae4e8b131218f942546b182e5418e119`
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

