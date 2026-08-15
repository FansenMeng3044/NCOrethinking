# Multi-vehicle CVRPTW for AM, POMO, and Split

This implementation uses one shared environment definition for four policies:

| Policy | Construction | Policy node features |
|---|---|---|
| AM-TW | Direct feasible routes | xy, demand, service, ready, due |
| POMO-TW | Direct feasible routes | xy, demand, service, ready, due |
| AM-Split-TW | Giant tour + exact hard Split | xy only |
| POMO-Split-TW | Giant tour + exact hard Split | xy only |

## Semantics

The decoder writes depot-delimited routes sequentially.  Every route represents
a different vehicle.  When depot `0` is selected, capacity is refilled and the
route clock is reset to `depot_start`.  The routes are independent in problem
time and may run in parallel even though the neural decoder writes them one by
one.

For customer `j`:

```text
arrival_j       = current_time + distance(i, j) / speed
service_start_j = max(arrival_j, tw_start_j)
completion_j    = service_start_j + service_time_j
```

An action is feasible only when service starts by `tw_end_j`, capacity is not
exceeded, and the vehicle can return to the depot by `depot_end`.  Waiting and
service time constrain feasibility but are not part of the score.  Reward is
negative total Euclidean distance over all vehicle routes.

`CVRPTWCore.py` is the source of truth for random data, geometric augmentation,
hard-TW Split, route reconstruction, and strict replay.  Both AM and POMO call
that implementation.

## Hard-TW Split

The Split decoder keeps the policy's customer permutation fixed.  For every
contiguous subsequence `tour[start:end+1]`, it simulates one fresh vehicle from
`depot_start`, including travel, waiting and service.  The segment is admitted
only when its total demand fits the vehicle, every service starts by the
customer due time, and the vehicle returns by `depot_end`.

Every feasible segment becomes an edge in an acyclic auxiliary graph.  Dynamic
programming finds a depot-to-end shortest path whose edge weight is
`depot -> first customer + internal customer distances + last customer -> depot`.
Thus the current objective is minimum total travel distance subject to hard
capacity/TW constraints.  Vehicle count, waiting, service and completion time
are not objective terms; route count is reported separately.

## Fixed dataset

Generate one dataset and use the same file for every model:

```bash
python NEW_py_ver/CVRP/generate_cvrptw_data.py \
  reproduction_data/fixed_testsets/cvrptw100_n10000_seed1234.pt \
  --problem-size 100 --samples 10000 --seed 1234
```

The canonical `.pt` keys are `depot_xy`, `node_xy`, `node_demand`,
`node_service_time`, `node_tw_start`, and `node_tw_end`.  Demands are normalized
to vehicle capacity 1.0.

## Training

```bash
# POMO-TW
python NEW_py_ver/CVRP/POMO_TW/train_n100.py

# XY-only POMO-Split-TW
python NEW_py_ver/CVRP/POMO_SPLIT_TW/train_n100.py

# AM-TW
python NEW_py_ver/CVRP/AM_SPLIT/train_am_tw.py

# XY-only AM-Split-TW
python NEW_py_ver/CVRP/AM_SPLIT/train_am_split_tw.py
```

Every entry point supports `--smoke` for a small CPU run.  Existing CVRP
checkpoints are not shape-compatible with TW-aware AM/POMO models and must not
be used as CVRPTW checkpoints.

All four TW training entry points retain the structured training recorder used
by their original CVRP counterparts.  Every run writes `training_metrics.csv`
inside its run directory.  It contains run metadata, batch rows, epoch
aggregates, checkpoint events, timing, throughput, GPU memory, losses, costs,
gradient norms and the final status row.  Use `--metrics-log-interval` to thin
batch rows without changing epoch aggregation and `--metrics-flush-interval`
to control disk flushing.  AM-TW and AM-Split-TW can be plotted directly with
`AM_SPLIT/plot_training_metrics.py`; POMO-TW and POMO-Split-TW use the same
POMO CSV schema as the original POMO and POMO-Split trainers.

## Evaluation

POMO entry points accept the same canonical dataset and augmentation 1 or 8:

```bash
python NEW_py_ver/CVRP/POMO_TW/test_n100.py DATASET \
  --checkpoint-dir RUN_DIR --epoch 200 --augmentation 8

python NEW_py_ver/CVRP/POMO_SPLIT_TW/test_n100.py DATASET \
  --checkpoint-dir RUN_DIR --epoch 200 --augmentation 8 \
  --output checkpoint_result.json
```

POMO-Split-TW selects the best `(augmentation, POMO start)` candidate, rebuilds
its explicit depot-delimited routes from the Split predecessor chain, and
strictly replays only that selected solution.  Its JSON includes 1x/8x mean
distance and standard error, mean vehicle count, replay counts and the overall
feasibility flag.

AM-TW and AM-Split-TW share an evaluator with strict replay and vehicle-count
reporting:

```bash
python NEW_py_ver/CVRP/AM_SPLIT/eval_cvrptw.py DATASET \
  --model RUN_DIR --augmentation 8 --strategy greedy
```

The reported score is always mean total travel distance, never elapsed time.

## Verification

```bash
python -m unittest discover -s NEW_py_ver/CVRP/tests -p "test_cvrptw.py" -v
```

The tests cover clock reset at depot, capacity refill, hard time-window Split,
strict infeasibility detection, feature isolation for Split, decoding, and
backpropagation for both AM variants.
