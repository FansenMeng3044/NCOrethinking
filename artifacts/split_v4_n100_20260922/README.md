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

All four n=100 models are complete and included here.  Each training process
exited with code 0, and each final checkpoint was loaded on CPU and checked for
the expected saved epoch.

| Model | Final checkpoint | Metric rows | Saved epoch |
|---|---|---:|---:|
| AM-Split CVRP100 | `checkpoints/am_split_cvrp100/epoch-100.pt` | 250,212 | 100 |
| AM-Split CVRPTW100 | `checkpoints/am_split_cvrptw100/epoch-100.pt` | 250,213 | 100 |
| POMO-Split CVRP100 | `checkpoints/pomo_split_cvrp100/checkpoint-2000.pt` | 318,012 | 2,000 |
| POMO-Split CVRPTW100 | `checkpoints/pomo_split_cvrptw100/checkpoint-2000.pt` | 318,012 | 2,000 |

All parsed numeric values in the four training metric files are finite.  The
training archives retain the complete metric CSV, training log, status files,
run metadata, and final checkpoint.

## Completed evaluations

The `evaluation` directory contains full result archives, saved route records,
verification outputs, logs, and exit-code files for every evaluation that is
already complete:

- AM-Split CVRP100: fixed CVRP100, CVRPLIB XML100, and size generalization to
  200, 500, and 1,000 customers;
- POMO-Split CVRP100: fixed CVRP100, CVRPLIB XML100, and the same size
  generalization sets; and
- POMO-Split CVRPTW100: fixed CVRPTW100 and Solomon-100.
- AM-Split CVRPTW100: fixed CVRPTW100, Solomon-100, XML100, and size
  generalization to 200, 500, and 1,000 customers.  XML100 and the larger
  sizes are explicitly labeled cross-constraint evaluations because the
  CVRPTW checkpoints receive neutral, inactive temporal attributes on CVRP
  instances.

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
| AM, fixed CVRPTW100 | 28.0958 | 30.5712 | +8.810% |
| AM, XML100 | 18,391.55 | 18,890.54 | +2.713% |
| POMO, XML100 | 17,926.07 | 18,597.31 | +3.744% |

For Solomon-100, the distance-first protocol gives 1,646.48 for POMO Direct
and 1,451.98 for POMO-Split, while the mean vehicle counts are 13.48 and 14.75,
respectively.  These figures are not the SINTEF lexicographic protocol and must
not be reported as official vehicle-first benchmark gaps.

For AM-CVRPTW100 on Solomon-100, Direct and Split obtain mean distances
1,695.87 and 1,796.25, with 9.96 and 15.30 vehicles.  Both solve all 56
instances.  On the fixed 10,000-instance CVRPTW100 set, Direct wins 9,909
paired instances and Split wins 91.  The AM-CVRPTW100 XML100 and
200/500/1,000-customer results are kept separate from native AM-CVRP results:
the Direct CVRPTW policy degenerates to one route per customer after removal of
active time constraints, whereas Split still partitions its customer order.
The full explanation and paired results are in
`reports/AM_SPLIT_CVRPTW100_FINAL_REPORT.md`.

The AM-CVRPTW100 evaluation archive preserves two superseded nonzero wrapper
records.  The first resulted from invoking the Solomon summarizer with an old
CLI, and the second from invoking the XML100 verifier from the old source
directory.  Both were corrected without changing any result row; the final
Solomon verification, XML100 verification, all six generalization stages, and
the aggregate validation all have exit code 0.

`CHECKSUMS.sha256` covers every committed binary archive and checkpoint.
