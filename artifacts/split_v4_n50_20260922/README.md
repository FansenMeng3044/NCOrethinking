# Split v4, n=50: code, final models, training records, and evaluation

This directory freezes the four final `n=50` Split models trained on 2026-09-21 and the complete evaluation completed on 2026-09-22. The source files committed with this artifact are the exact files deployed for these runs.

## Model definition

The Split policies use the same static customer features and dynamic decoder context as their matched Direct policies:

- CVRP customer encoder input: `(x, y, demand)`; decoder context: current-node embedding and remaining load.
- CVRPTW customer encoder input: `(x, y, demand, service time, ready time, due time)`; decoder context: current-node embedding, remaining load, and current time.

The policy action space remains customer-only. The depot is encoded but is never selected as an action, and capacity/time-window violations are not added to the action mask. The exact Split decoder constructs the feasible routes after the complete customer permutation has been generated.

## Final checkpoints

| Model | Final model | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| AM-Split CVRP50 | `checkpoints/am_split_cvrp50/epoch-100.pt` | 17,273,995 | `ff3c9fe072f56d4e888aed8eceef14d8b023e2aef4f04abd1c35567f5bb87279` |
| POMO-Split CVRP50 | `checkpoints/pomo_split_cvrp50/checkpoint-2000.pt` | 15,236,311 | `56f608725a5cbf1c1f089da9f2ead35ccd677abed041ffd7b46ffaa6b5f20059` |
| AM-Split CVRPTW50 | `checkpoints/am_split_cvrptw50/epoch-100.pt` | 23,283,083 | `77eae73a164c9a58e74329b874cf6a8ed2e427318a28aa8821cb91aeaee0a1f1` |
| POMO-Split CVRPTW50 | `checkpoints/pomo_split_cvrptw50/checkpoint-2000.pt` | 15,242,583 | `d252479b9fa91981cf9baad84a2880ecd6076330d6ea2b2e5de9e4b499134b98` |

All four files were loaded on CPU with `torch.load(..., weights_only=False)`. AM checkpoints contain the model, optimizer, baseline, epoch, and RNG states. POMO checkpoints contain the model, optimizer, scheduler, environment/model parameters, epoch, and result log.

## Training records

The exact `training_metrics.csv` files are stored losslessly as `training_metrics.csv.tar.gz`:

| Model | CSV rows | Uncompressed bytes | Non-finite numeric values |
|---|---:|---:|---:|
| AM-Split CVRP50 | 250,212 | 123,101,803 | 0 |
| POMO-Split CVRP50 | 318,012 | 131,224,059 | 0 |
| AM-Split CVRPTW50 | 250,212 | 123,412,902 | 0 |
| POMO-Split CVRPTW50 | 318,012 | 130,780,982 | 0 |

The AM argument files and the available POMO run log are stored beside the metric archives.

## Evaluation records

The expanded `evaluation/` directory contains:

- fixed CVRP50 results: 10,000 instances for each of AM Direct, AM Split, POMO Direct, and POMO Split;
- fixed CVRPTW50 results: 10,000 instances for each of the same four variants;
- CVRPLIB XML100: 40,000 model-instance records, summaries, manifest, and independent verification report;
- Solomon-50: 224 model-instance records, summary, family comparison, and independent verification report;
- evaluation scripts, model manifests, smoke-test outputs, run logs, and exit-status files.

`evaluation/evaluation_complete_with_routes.tar.gz` is the complete server-side evaluation directory and additionally contains every saved XML100 and Solomon route JSON. It can be extracted with:

```bash
tar -xzf evaluation/evaluation_complete_with_routes.tar.gz
```

Both evaluation lanes exited with code `0`; the saved phase is `completed`. The XML100 verifier accepted all 40,000 routes and the Solomon verifier accepted all 224 routes.

## Main results

All comparisons below use the matched 8-fold protocol and report Split relative to Direct.

| Evaluation | AM Direct | AM Split | AM change | POMO Direct | POMO Split | POMO change |
|---|---:|---:|---:|---:|---:|---:|
| Fixed CVRP50 mean cost | 10.6905 | 11.1037 | +3.87% | 10.4558 | 10.6989 | +2.32% |
| Fixed CVRPTW50 mean distance | 16.0429 | 17.0127 | +6.05% | 15.0715 | 15.7631 | +4.59% |
| XML100 mean cost | 18,287.14 | 18,965.14 | +3.71% | 18,018.69 | 18,826.60 | +4.48% |
| Solomon-50 mean distance | 844.89 | 927.67 | +9.80% | 753.97 | 789.44 | +4.71% |

For the complete results, use `evaluation/results/comparison_report.json`, `comparison_overall.csv`, `comparison_paired.csv`, and the benchmark-specific tables.

## Integrity

`CHECKSUMS.sha256` lists the repository artifact checksums. `REMOTE_EXPORT_SHA256SUMS.txt` records the checksums of the original server export archives.
