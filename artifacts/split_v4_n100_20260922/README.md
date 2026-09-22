# Split v4 n=100 artifacts

This directory records the corrected Split v4 implementation and the completed
training and evaluation artifacts for models trained with 100 customers.  The
implementation is the same corrected code committed with the n=50 artifact:

- the static encoder inputs match the corresponding Direct model;
- the decoder keeps the Direct dynamic context;
- the depot is not a customer-ordering action;
- capacity and time-window feasibility masks are not applied while producing
  the customer permutation; and
- the exact Split result supplies the terminal training objective.

## Completion status

Three n=100 models are complete and included here.  Each training process
exited with code 0, and each final checkpoint was loaded on CPU and checked for
the expected saved epoch.

| Model | Final checkpoint | Metric rows | Saved epoch |
|---|---|---:|---:|
| AM-Split CVRP100 | `checkpoints/am_split_cvrp100/epoch-100.pt` | 250,212 | 100 |
| POMO-Split CVRP100 | `checkpoints/pomo_split_cvrp100/checkpoint-2000.pt` | 318,012 | 2,000 |
| POMO-Split CVRPTW100 | `checkpoints/pomo_split_cvrptw100/checkpoint-2000.pt` | 318,012 | 2,000 |

All parsed numeric values in the three training metric files are finite.  The
training archives retain the complete metric CSV, training log, status files,
run metadata, and final checkpoint.

AM-Split CVRPTW100 was still training when this initial artifact commit was
made.  Its final epoch-100 checkpoint, training archive, and matched completed
evaluations must be added only after the process exits successfully.  No
intermediate checkpoint is represented as a final model here.

## Completed evaluations

The `evaluation` directory contains full result archives, saved route records,
verification outputs, logs, and exit-code files for every evaluation that is
already complete:

- AM-Split CVRP100: fixed CVRP100, CVRPLIB XML100, and size generalization to
  200, 500, and 1,000 customers;
- POMO-Split CVRP100: fixed CVRP100, CVRPLIB XML100, and the same size
  generalization sets; and
- POMO-Split CVRPTW100: fixed CVRPTW100 and Solomon-100.

All formal evaluation lanes in these archives have exit code 0.  The saved
routes were independently replayed under the corresponding capacity and, where
applicable, hard time-window constraints.  Concise result tables are available
under `reports`.

## Main completed results

| Model and evaluation | Direct | Split | Relative Split change |
|---|---:|---:|---:|
| AM, fixed CVRP100 | 16.3084 | 17.2217 | +5.600% |
| POMO, fixed CVRP100 | 15.8511 | 16.5395 | +4.343% |
| POMO, fixed CVRPTW100 | 25.7140 | 27.9577 | +8.725% |
| AM, XML100 | 18,391.55 | 18,890.54 | +2.713% |
| POMO, XML100 | 17,926.07 | 18,597.31 | +3.744% |

For Solomon-100, the distance-first protocol gives 1,646.48 for POMO Direct
and 1,451.98 for POMO-Split, while the mean vehicle counts are 13.48 and 14.75,
respectively.  These figures are not the SINTEF lexicographic protocol and must
not be reported as official vehicle-first benchmark gaps.

`CHECKSUMS.sha256` covers every committed binary archive and checkpoint.
