# POMO-MTL-Split n=50 (B-mask v5)

This directory archives the final training and official MVMoE evaluation artifacts for the corrected `POMO-MTL-Split` model trained at size 50.

## Source and training protocol

- Repository source commit: `5eb60377cb847ba68b88b24b960b9bbc76fdf286`
- Official Routing-MVMoE source commit: `af29e5af0595f94f3ecc3bc46d72df1089a62682`
- Model: `MTL_SPLIT`
- Training problem set: `Train_ALL` (the six official training variants)
- Problem size / POMO size: 50 / 50
- Epochs: 5,000
- Training episodes per epoch: 20,000
- Batch size: 128
- Seed: 2023
- Learning rate at completion: `1e-5`
- Split backend: fused Triton implementation
- Checkpoint interval: 300 epochs, plus the final epoch

The exact B-mask v5 source used by this run is tracked in
`third_party/Routing-MVMoE-overlay`.  The ten trajectory-affecting files listed
in the checkpoint's `source_fingerprint` match that overlay byte for byte after
checkout on Linux.  The recorded outer commit identifies the repository base;
the overlay in this artifact branch records the previously uncommitted source
that was deployed for training.

The decoder keeps only depot/visited action validity and the official backhaul-order feasibility mask. Capacity, time-window, open-route, and route-duration constraints are not used as action masks; they are enforced by Split when the completed customer order is partitioned.

## Validation

- Training wrapper exit code: 0
- Final checkpoint epoch: 5,000
- `torch.load(..., map_location="cpu", weights_only=False)` succeeded.
- The checkpoint contains model, optimizer, scheduler, RNG, source-fingerprint, and strict-resume state.
- `training_metrics.csv` contains 795,019 data rows and ends with `run_summary,run_finished`.
- All checked applicable values in `train_score`, `train_loss`, `learning_rate`, `grad_norm`, `step_seconds`, and `throughput` are finite.
- Applying the archived overlay to official Routing-MVMoE commit
  `af29e5af0595f94f3ecc3bc46d72df1089a62682` passes the complete Split test
  suite: 152 tests passed and 82 CUDA-specific tests were skipped on the local
  CPU verification host.

## Evaluation protocol

The completed size-50 checkpoint is evaluated with the official Routing-MVMoE instance generator and evaluator on all 16 variants, with 1,000 generated instances per variant. Evaluation is reported at sizes 50, 100, and 200. The Direct comparison uses the official POMO-MTL checkpoints under the same generated-instance protocol.

The final evaluation status is `completed` with exit code 0. Earlier failed attempts are retained in `evaluation/logs` and `evaluation/status`: attempt 0 failed before evaluation because of a missing shell command, and the size-200 Split stage was rerun with a smaller batch after a GPU-memory failure. No result rows from failed partial attempts replace completed outputs.

## Aggregate comparison

| Test size | Regime | Paired instances | Direct mean cost | Split mean cost | Split vs. Direct |
|---:|:---|---:|---:|---:|---:|
| 50 | all | 12,926 | 9.4500 | 9.8022 | +3.727% |
| 50 | trained variants | 6,000 | 9.9701 | 10.2517 | +2.824% |
| 50 | unseen variants | 6,926 | 8.9995 | 9.4128 | +4.594% |
| 100 | all | 12,178 | 15.1548 | 16.4381 | +8.467% |
| 100 | trained variants | 6,000 | 16.2880 | 17.5549 | +7.778% |
| 100 | unseen variants | 6,178 | 14.0544 | 15.3534 | +9.243% |
| 200 | all | 12,004 | 26.7997 | 28.9560 | +8.046% |
| 200 | trained variants | 6,000 | 28.6471 | 30.5699 | +6.712% |
| 200 | unseen variants | 6,004 | 24.9536 | 27.3433 | +9.577% |

Paired means use only instances solved by both methods. The four unseen variants combining backhauls and hard time windows do not always admit a feasible contiguous partition of the predicted customer order; their failures are recorded explicitly in `evaluation/results/comparison_by_environment.csv`.

## Files

- `training/epoch-5000.pt`: final checkpoint (Git LFS)
- `training/epoch-5000.pt.resume.json`: strict-resume metadata
- `training/training_metrics.csv.zst`: lossless Zstandard-compressed training CSV (Git LFS)
- `training/train.log`, `training/run.sh`, `training/status/`: complete training log, launch command, and terminal status
- `training/deployment_manifest.json`: source and masking contract
- `evaluation/results/`: per-instance Direct and Split results plus aggregate tables
- `evaluation/logs/`, `evaluation/status/`: all evaluation attempts and final status
- `MANIFEST.sha256`: SHA-256 hashes for all archived files
