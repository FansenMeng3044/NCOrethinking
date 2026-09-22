# POMO-MTL-Split n=100 (B-mask v5)

This directory archives the final training and official MVMoE evaluation artifacts for the corrected `POMO-MTL-Split` model trained at size 100.

## Source and training protocol

- Repository source commit: `5eb60377cb847ba68b88b24b960b9bbc76fdf286`
- Official Routing-MVMoE source commit: `af29e5af0595f94f3ecc3bc46d72df1089a62682`
- Model: `MTL_SPLIT`
- Training problem set: `Train_ALL` (the six official training variants)
- Problem size / POMO size: 100 / 100
- Epochs: 5,000
- Training episodes per epoch: 20,000
- Global / per-rank batch size: 256 / 64
- DDP world size: 4
- Seed: 2023
- Initial learning rate: `1e-4`; scheduled learning rate after epoch 4,500: `1e-5`
- Split backend: fused Triton implementation
- Checkpoint interval: 300 epochs, plus the final epoch

The exact B-mask v5 source used by this run is tracked in
`third_party/Routing-MVMoE-overlay`. The ten trajectory-affecting files listed
in the checkpoint's `source_fingerprint` match that overlay byte for byte. The
recorded outer commit identifies the repository base; the overlay records the
previously uncommitted source that was deployed for training.

The static encoder input is the original MVMoE representation: depot
coordinates and customer coordinates, demand, time-window start, and
time-window end. The decoder receives load, current time, route length, and
open-route state. Its action mask contains only depot/visited validity and the
official backhaul-order feasibility rule. Capacity, time-window, open-route,
and route-duration constraints are enforced by Split after construction of the
customer order.

## Validation

- Training wrapper exit code: 0
- Final checkpoint epoch: 5,000
- Final checkpoint SHA-256: `85c8dcc12aeca7cd3a99684a44329f5eab522c14aa5cf329416269b0fa0653cb`
- `torch.load(..., map_location="cpu", weights_only=False)` succeeded.
- The checkpoint contains model, optimizer, scheduler, RNG, source-fingerprint, and strict-resume state.
- `training_metrics.csv` contains 405,019 data rows, records 100,000,000 training samples, and ends with `run_summary,run_finished`.
- All applicable numerical values in the training CSV are finite.
- The measured wall-clock training time is 85,815.25 seconds (23.84 hours).
- Applying the archived overlay to the official Routing-MVMoE source passes the complete Split test suite: 152 tests passed and 82 CUDA-specific tests were skipped on the local CPU verification host.

## Evaluation protocol

The completed size-100 checkpoint is evaluated on the official MVMoE `.pkl`
files for all 16 routing variants, with 1,000 fixed instances per variant. The
same trained checkpoint is evaluated at problem sizes 50, 100, and 200. Both
the Split model and the official Direct POMO-MTL reference use greedy POMO
decoding with `pomo_size=n` and 8-fold geometric augmentation. The evaluation
scripts independently replay each route returned by Split and require the
replayed objective value to agree with the Split value.

All three evaluation stages completed with exit code 0. Direct solved all
48,000 model-instance pairs. Split solved every instance from the six variants
represented during training. Its unsuccessful cases occur in unseen variants
that combine backhauls and hard time windows, where the predicted customer
order has no feasible contiguous partition.

## Aggregate comparison

| Test size | Regime | Paired instances | Direct mean cost | Split mean cost | Split vs. Direct |
|---:|:---|---:|---:|---:|---:|
| 50 | all | 12,700 | 9.6228 | 10.3080 | +7.121% |
| 50 | trained variants | 6,000 | 10.1534 | 10.7475 | +5.851% |
| 50 | unseen variants | 6,700 | 9.1475 | 9.9145 | +8.384% |
| 100 | all | 12,099 | 14.6369 | 15.4956 | +5.867% |
| 100 | trained variants | 6,000 | 15.7490 | 16.6199 | +5.530% |
| 100 | unseen variants | 6,099 | 13.5429 | 14.3896 | +6.252% |
| 200 | all | 12,002 | 23.5044 | 26.0671 | +10.903% |
| 200 | trained variants | 6,000 | 25.0208 | 27.6868 | +10.655% |
| 200 | unseen variants | 6,002 | 21.9885 | 24.4480 | +11.185% |

Paired means use only instances solved by both methods. Per-variant success
counts, costs, and vehicle counts are retained in
`evaluation/results/comparison_by_environment.csv`.

## Files

- `training/epoch-5000.pt`: final checkpoint (Git LFS)
- `training/epoch-5000.pt.resume.json`: strict-resume metadata
- `training/training_metrics.csv.zst`: lossless Zstandard-compressed training CSV (Git LFS)
- `training/train.log`, `training/run.sh`, `training/status/`: complete training log, launch command, and terminal status
- `training/deployment_manifest.json`: source and masking contract
- `evaluation/results/`: per-instance Direct and Split results plus aggregate tables
- `evaluation/logs/`, `evaluation/status/`: evaluation logs and terminal status for all three sizes
- `evaluation/scripts/`: exact evaluation and summarization entry points
- `MANIFEST.sha256`: SHA-256 hashes for every archived file except the manifest itself
