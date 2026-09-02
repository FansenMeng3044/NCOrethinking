# Strict XY-only giant-tour variants

This extension adds three models without editing the official MVMoE models or
environments:

- `MTL_SPLIT`: POMO-MTL-Split;
- `MOE_SPLIT`: MVMoE/4E-Split;
- `MOE_LIGHT_SPLIT`: MVMoE/4E-L-Split.

## Experimental boundary

The policy receives exactly `depot_xy` and `node_xy`. Customer embeddings have
input dimension two. The direct decoder's dynamic
`[load, current_time, length, open]` input is absent. The depot is permanently
masked and the only other action mask is the visited-customer mask. Every
complete rollout is therefore a customer permutation.

`MVMoEInstanceAdapter` creates two disjoint views of an untouched official
environment. `PolicyView` contains only coordinates. `ConstraintSpec` contains
the normalized demand/capacity, open-route flag, signed backhaul demand, route
limit, service durations, time windows, depot horizon, speed, and objective
rounding configuration. Only the exact generalized Split decoder receives the
latter.

The Split decoder implements all 16 official combinations compositionally. It
uses MVMoE's exact semantics: signed load stays in `[0,1]` for backhauls; a
backhaul route starts full while any linehaul remains and empty afterwards;
open routes omit the return edge and return constraints; length and temporal
feasibility use unrounded Euclidean distance; objective edges alone use
`loc_scaler` rounding when configured.

## Training

The official six-task mixture is unchanged:

```bash
python train_split.py --problem Train_ALL --model_type MTL_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_LIGHT_SPLIT --problem_size 100 --pomo_size 100
```

Reward is negative post-Split distance. Infeasible giant tours remain explicitly
infeasible; they are never relaxed or repaired. The POMO baseline and policy
loss are computed only over finite candidates, and training stops with a clear
error if an instance has no feasible candidate. MoE auxiliary losses are kept.

The official schedule is retained: 5,000 epochs, 20,000 generated instances per
epoch, batch size 128, Adam at `1e-4`, a `0.1` learning-rate decay at milestone
4,501, and checkpoints at epochs 2,500 and 5,000. Each run also writes
`training_metrics.csv` beside its checkpoints. The file contains run metadata,
batch and epoch metrics, checkpoint events, and a final run summary. Batch rows
record the sampled task, learning rate, score/cost and loss statistics, policy
and MoE auxiliary losses, gradient norm, Split-feasible candidate rate, mean
route count, timing, throughput, and GPU memory. `--metrics_log_interval N`
retains every Nth batch row; epoch, checkpoint, and summary rows are always kept.

## Evaluation on all 16 environments

```bash
python test_split.py --checkpoint PATH/epoch-5000.pt --model_type MOE_SPLIT \
  --problem ALL --problem_size 100 --augmentation 8 --output results.csv
```

Evaluation performs an independent double-precision route replay before a row
is accepted. It checks exact coverage, duplicates, capacity/backhaul load,
route length, customer time windows, depot return, open-route cost, and total
distance.

## Verification

```bash
python -m pytest tests_split -q
```

The suite compares dynamic programming with exhaustive boundary enumeration on
all 16 combinations (with and without objective rounding), adapts every
official environment, audits the policy state/action mask, verifies identical
tours for identical coordinates under different constraints, and runs finite
forward/backward checks for all three models on all six training tasks.
