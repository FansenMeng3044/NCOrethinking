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

## Evaluation

POMO entry points accept the same canonical dataset and augmentation 1 or 8:

```bash
python NEW_py_ver/CVRP/POMO_TW/test_n100.py DATASET \
  --checkpoint-dir RUN_DIR --epoch 200 --augmentation 8

python NEW_py_ver/CVRP/POMO_SPLIT_TW/test_n100.py DATASET \
  --checkpoint-dir RUN_DIR --epoch 200 --augmentation 8
```

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
