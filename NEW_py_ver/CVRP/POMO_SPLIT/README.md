# POMO giant tour + Split for CVRP

This experiment separates customer sequencing from capacity decisions:

1. The POMO policy sees depot coordinates and customer coordinates plus demand, and selects each customer exactly once.
2. Depot is encoder context only; it is never an action.
3. The decoder receives the same dynamic resource context as Direct POMO:
   the current-node embedding and remaining capacity, plus current time for
   CVRPTW. Capacity and time-window violations are deliberately excluded from
   the action mask; only the depot and already visited customers are masked.
   Because a giant tour contains no depot actions, these resource values are
   propagated along the full permutation and are not reset before Split.
4. At the terminal step, hard-capacity Split optimally partitions the fixed giant tour.
5. The policy-gradient reward is the negative decoded CVRP distance.

For one POMO rollout:

```text
giant tour: [4, 2, 1, 3]
Split:      [4, 2] | [1, 3]
reward:     -(d(0,4)+d(4,2)+d(2,0)+d(0,1)+d(1,3)+d(3,0))
```

The decoder uses the exact Bellman shortest-path formulation, vectorized over
all batch and POMO tours. It does not reorder customers.

Run a CPU smoke test:

```console
python train_n100.py --smoke
```

Start the configured CVRP100 training run:

```console
python train_n100.py
```
