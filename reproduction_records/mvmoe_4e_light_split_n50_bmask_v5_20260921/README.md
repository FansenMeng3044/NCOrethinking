# MVMoE/4E-L-Split n=50 reproducibility record

This directory archives the completed B-mask v5 training run and its official
MVMoE evaluations. The policy uses the original five-dimensional static node
features and the original dynamic decoder context. Only the backhaul ordering
feasibility mask is applied during permutation construction. Capacity, time
windows, backhauls, open routes, and route length are checked by Split.

## Training

- Model: `MOE_LIGHT_SPLIT`
- Problem size and POMO size: 50
- Epochs: 5,000
- Global training episodes per epoch: 20,000
- Seed: 2023
- Checkpoint interval: 300 epochs
- Final checkpoint: `training/epoch-5000.pt`
- Checkpoint schema: complete model, optimizer, scheduler, RNG, distributed,
  runtime, source, and resume state
- Training metrics: `training/training_metrics.csv.gz` (lossless gzip)
- Training exit code: 0

The metrics file contains 795,019 records, including 785,000 batch records and
5,000 epoch summaries. It reaches epoch 5,000, contains no non-finite values in
the applicable optimization fields, and records learning rates of 1e-4 and
1e-5 under the configured schedule.

## Evaluation

The evaluation uses the official fixed n=50, n=100, and n=200 datasets. Each
size contains 16 environments with 1,000 instances per environment. Inference
uses argmax decoding, eight-fold geometric augmentation, and POMO size equal to
the evaluated problem size. Direct and Split results contain 16,000 unique
environment-instance rows per size.

Split successes were 12,897/16,000 at n=50, 12,189/16,000 at n=100, and
12,002/16,000 at n=200. All 6,000 instances from the six training variants were
solved at every size. Failures in unseen variants are retained as model results,
not removed or repaired. See `evaluation/results/comparison_summary.csv` and
`evaluation/results/comparison_by_environment.csv` for the paired comparison.

Both training and evaluation exit codes are zero. The CSV structure, unique
keys, finite values, and summary row counts were independently validated.
