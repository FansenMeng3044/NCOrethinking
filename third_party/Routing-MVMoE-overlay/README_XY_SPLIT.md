# Original-input giant-tour variants with a B-only feasibility mask

This extension adds three models without editing the official MVMoE models or
environments:

- `MTL_SPLIT`: POMO-MTL-Split;
- `MOE_SPLIT`: MVMoE/4E-Split;
- `MOE_LIGHT_SPLIT`: MVMoE/4E-L-Split.

## Constraint factorization

The static encoder matches the original POMO-MTL and MVMoE input contract. The
depot receives `(x, y)`, while every customer receives
`(x, y, demand, tw_start, tw_end)`. Non-TW environments set both TW fields to
zero. Service time is not a static encoder feature.

The decoder also matches the original dynamic input contract. Its query uses
the current customer embedding together with `(load, current_time, length,
open)`. The values are updated after every selected customer. Because the
policy never selects the depot, they evolve continuously along the giant tour
and are not reset at route boundaries during permutation construction.

The action mask always excludes the depot and customers already visited. In a
backhaul environment, it additionally excludes a customer when that customer
can neither continue the current hidden B-feasible route nor start a new
B-feasible route. This hidden state is only a feasibility witness and is not an
additional decoder feature. Capacity apart from the B signed-load semantics,
hard time windows (TW), open routes (O), and route-duration limits (L) never
exclude an action. The policy emits exactly one customer permutation. The exact
Split stage subsequently enforces all applicable C, TW, B, O, and L semantics
over its contiguous partitions and remains free to choose different boundaries.

`MVMoEInstanceAdapter` leaves every official environment unchanged. Its
`PolicyView` contains depot coordinates and the original five customer
features, while `ConstraintSpec` contains the complete instance used by the
dynamic context and final Split stage.

All feasibility checks use the official MVMoE environment tolerance
`round_error_epsilon = 1e-5`. The reference Split, fused Triton Split,
giant-tour environment, and independent route replay share this single
constant so generator acceptance and downstream validation agree at numerical
boundaries.

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

Reward is negative post-Split distance. No repair, sampling fallback, or
constraint relaxation is applied. The POMO baseline and policy loss are computed only
over finite candidates, and training stops with a clear error if an instance
has no feasible candidate. MoE auxiliary losses are kept.

The official optimization schedule is retained: 5,000 epochs, 20,000 generated
instances per epoch, batch size 128, Adam at `1e-4`, and a `0.1` learning-rate
decay at milestone 4,501. Operational checkpoints are written every 300
completed epochs and at epoch 5,000; checkpoint frequency does not alter the
optimization trajectory. Each run also writes
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
each rank. Every checkpoint has a `.resume.json` sidecar containing its SHA256,
completed/next epoch, full trajectory contract, code hashes, runtime versions,
and RNG-state inventory. Resume verifies the sidecar checksum, identical
trajectory-affecting source, model/environment/optimizer/scheduler settings,
Split backend and tolerance, batch/instance budgets, seed, and DDP world size
before loading. Consequently DDP changes
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
adapts every official environment at sizes 50 and 100, verifies the exact
five-feature encoder input and zero TW fields in non-TW tasks, verifies the
four original dynamic decoder attributes and constraint-free customer masks,
and runs finite forward/backward checks for all three models and both sizes on
all six training tasks.
