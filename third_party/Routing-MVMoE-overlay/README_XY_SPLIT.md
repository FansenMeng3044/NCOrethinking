# XY-encoded, B-aware giant-tour variants

This extension adds three models without editing the official MVMoE models or
environments:

- `MTL_SPLIT`: POMO-MTL-Split;
- `MOE_SPLIT`: MVMoE/4E-Split;
- `MOE_LIGHT_SPLIT`: MVMoE/4E-L-Split.

## Constraint factorization

The static encoder receives exactly `depot_xy` and `node_xy`; customer
embeddings therefore retain input dimension two. During autoregressive
decoding, only the backhaul (B) state is exposed. Route-length (L), ordinary
capacity (C), and time-window (TW) attributes never enter the neural model.

The decoder always emits exactly one permutation and never selects the depot.
For each candidate it evaluates both continuation of the current hidden B route
and restart from the depot. B uses MVMoE's signed-load transition, including
its normalized signed customer demand and full/empty restart rule. If
continuation is infeasible but restart is feasible, a hidden new route begins
automatically. These B starts form a feasibility witness; a customer infeasible
under both B transitions is masked. L never affects decoder features, candidate
scores, masks, or starts.

`MVMoEInstanceAdapter` leaves every official environment unchanged.  Its
`PolicyView` contains only coordinates for static encoding. B flags and dynamic
signed-load state are consumed during giant-tour decoding. The exact Split
stage independently enforces B, L, C, and TW over all contiguous partitions.
The decoder's hidden B boundaries are only a feasibility witness: Split may
move, add, or remove them when selecting the minimum-cost feasible partition.

The resulting pipeline implements all 16 official combinations
compositionally. Signed demands retain MVMoE's capacity accounting: a route
starts full while any linehaul remains in the suffix and empty once only
backhauls remain. Open routes omit return distance and depot-return timing.
L and TW use raw distance only in Split; objective edges alone use `loc_scaler`
rounding when configured.

## Training

The official six-task mixture is unchanged:

```bash
python train_split.py --problem Train_ALL --model_type MTL_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_SPLIT --problem_size 100 --pomo_size 100
python train_split.py --problem Train_ALL --model_type MOE_LIGHT_SPLIT --problem_size 100 --pomo_size 100
```

Reward is negative post-Split distance. The B construction is deterministic
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

### Two-GPU DDP

DDP treats `--train_batch_size` and `--train_episodes` as global quantities.
The controlled two-GPU configuration therefore uses a local batch of 64 on
each rank while retaining the official global batch of 128 and 20,000 global
instances per epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 train_split.py \
  --ddp --expected_world_size 2 --problem Train_ALL \
  --model_type MOE_SPLIT --problem_size 100 --pomo_size 100 \
  --train_episodes 20000 --train_batch_size 128
```

Each optimizer update combines two equal 64-instance shards. Rank 0 selects
and broadcasts the training task, so a global batch still contains one of the
six official environments. Input-choice MoE load-balancing statistics are
computed over the complete global batch, and the stochastic dense/MoE branch
of 4E-L is synchronized across ranks. Metrics are globally reduced; only rank
0 writes CSV files and atomic checkpoints. Checkpoints store the RNG state for
each rank and can be resumed with the same world size. Consequently DDP changes
only execution parallelism, not global batch size, instance budget, optimizer,
learning-rate schedule, reward, or constraint factorization.

On a six-GPU host, `../run_three_mvmoe_split_n100_ddp.sh` launches the three
size-100 models concurrently on GPU pairs `0,1`, `2,3`, and `4,5`.

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

The suite compares the B/L/C/TW dynamic program with exhaustive boundary
enumeration on all 16 combinations (with and without objective rounding),
adapts every official environment at sizes 50 and 100, audits B transition
masks and mandatory starts, verifies L/C/TW blindness under fixed coordinates,
and runs finite forward/backward checks for all three models and both sizes on
all six training tasks.
