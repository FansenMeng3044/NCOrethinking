# XY-encoded, B/L-aware giant-tour variants

This extension adds three models without editing the official MVMoE models or
environments:

- `MTL_SPLIT`: POMO-MTL-Split;
- `MOE_SPLIT`: MVMoE/4E-Split;
- `MOE_LIGHT_SPLIT`: MVMoE/4E-L-Split.

## Constraint factorization

The static encoder receives exactly `depot_xy` and `node_xy`; customer
embeddings therefore retain input dimension two.  During autoregressive
decoding, only the backhaul (B) and route-length (L) state is exposed.  Capacity
(C) and time-window (TW) attributes never enter the neural model.

The decoder always emits exactly one permutation and never selects the depot.
For each candidate it evaluates both continuation of the current hidden route
and restart from the depot.  B uses MVMoE's signed-load transition, including
its full/empty restart rule; L uses the official open- or closed-route length
test.  If continuation is infeasible but restart is feasible, a hidden new
route begins automatically.  These starts are saved as mandatory boundaries;
a customer infeasible under both transitions is masked.

`MVMoEInstanceAdapter` leaves every official environment unchanged.  Its
`PolicyView` contains only coordinates for static encoding.  B/L flags and
their dynamic state are consumed only during giant-tour decoding.  The exact
Split stage can add route boundaries for C/TW, but cannot merge across the
decoder's mandatory B/L boundaries.

The resulting pipeline implements all 16 official combinations
compositionally.  Signed demands retain MVMoE's capacity accounting: a route
starts full while any linehaul remains in the suffix and empty once only
backhauls remain.  Open routes omit return distance and depot-return timing.
L uses raw Euclidean distance in the decoder; TW uses raw distance in Split;
objective edges alone use `loc_scaler` rounding when configured.

## Training

The official six-task mixture is unchanged:

```bash
python train_split.py --problem Train_ALL --model_type MTL_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_LIGHT_SPLIT --problem_size 100 --pomo_size 100
```

Reward is negative post-Split distance. The B/L construction is deterministic
conditional on the selected order; no repair, sampling fallback, or constraint
relaxation is applied.  The POMO baseline and policy loss are computed only
over finite candidates, and training stops with a clear error if an instance
has no feasible candidate. MoE auxiliary losses are kept.

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

The suite compares the restricted C/TW dynamic program with exhaustive boundary
enumeration on all 16 combinations (with and without objective rounding),
adapts every official environment at sizes 50 and 100, audits B/L transition
masks and mandatory starts, verifies C/TW blindness under fixed coordinates, and runs
finite forward/backward checks for all three models and both sizes on all six
training tasks.
