# Preliminary audit: POMO giant-tour and Split

Date: 2026-07-13

## Scope and evidence boundary

This report audits the public remote `main` tree at commit
`15338053a7b7e3e03f85b3d4863f29a21bea4812`. It does **not** yet prove which
source snapshot or checkpoint produced the senior's poor result. The exact
run's `result/.../src` snapshot and `run_log.txt` remain authoritative for that
question.

Sources reviewed:

- Jia, Mei, Zhang (2022), *Confidence-Based Ant Colony Optimization for
  Capacitated Electric Vehicle Routing Problem With Comparison of Different
  Encoding Schemes*.
- Prins (2004), *A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem*, cited as reference [29] by Jia et al.
- Current repository modules `POMO_SPLIT`, `POMO_SPLIT_EVALONLY`, and
  `POMO_TOUR`.

## The reference Split algorithm

For a fixed customer permutation `pi[0:n]`, classical Split creates auxiliary
vertices `0..n`. An auxiliary arc `(i,j)` represents one route

```text
depot -> pi[i] -> ... -> pi[j-1] -> depot
```

when its total demand does not exceed vehicle capacity. The arc weight is that
route's complete distance. A shortest path from auxiliary vertex `0` to `n`
therefore gives the minimum-total-distance capacity-feasible contiguous
partition of the fixed order. Split chooses boundaries only; it does not reorder
customers.

Jia et al. describe their indirect encoding as a giant tour followed by an
optimal dynamic-programming Split, and cite Prins [29]. Their CEVRP battery and
charging subproblem is separate and must not be added to the ordinary POMO
CVRP100 Split.

References:

- https://openaccess.wgtn.ac.nz/articles/journal_contribution/Confidence-based_Ant_Colony_Optimization_for_Capacitated_Electric_Vehicle_Routing_Problem_with_Comparison_of_Different_Encoding_Schemes/19704493
- https://doi.org/10.1016/S0305-0548(03)00158-8
- https://doi.org/10.1016/j.trc.2014.01.011

## Current experiment variants

| Module | Training reward | Evaluation objective | Demand/capacity visible to policy |
|---|---|---|---|
| `POMO_SPLIT` | Negative optimal post-Split CVRP distance | Post-Split CVRP distance | No |
| `POMO_SPLIT_EVALONLY` | Negative uncapacitated `depot -> all customers -> depot` distance | Post-Split CVRP distance | No |
| `POMO_TOUR` | Raw `single_route` or customer-only `tsp_cycle` distance | Must be traced through the actual run/evaluator | No |

All inspected models encode depot coordinates as context. Depot is masked from
the action space, but it is not absent from the neural input. The missing
instance information is customer demand (and an explicit capacity value).

## Code-to-reference comparison

| Reference requirement | Current implementation evidence | Preliminary assessment |
|---|---|---|
| Customer IDs form a full permutation | `SplitDecoder._validate_inputs`, lines 149-182 | Aligned |
| IDs `1..n` map to coordinate/demand indices `0..n-1` | lines 50-57 | Aligned |
| Segment demand is computed for every contiguous `pi[i:j]` | lines 72-76 and 90-100 | Aligned |
| Capacity feasibility is `load <= Q + epsilon` | line 100 | Aligned |
| Route cost includes depot-out, all internal edges, depot-return | lines 59-70 and 89-97 | Aligned |
| Every feasible predecessor is considered | lines 90-104 | Aligned; this is dynamic programming, not greedy filling |
| Bellman label starts at zero and minimizes total route cost | lines 78-106 | Aligned |
| Predecessors recover contiguous routes | lines 119-146 | Aligned |
| Objective is total route length | `final_cost = potential[:, problem_size]` | Aligned |
| No fixed route count or maximum route-length constraint | no such DP state/mask | Correct for official POMO CVRP100 |

Pinned implementation:

https://github.com/FansenMeng3044/pomo/blob/15338053a7b7e3e03f85b3d4863f29a21bea4812/NEW_py_ver/CVRP/POMO_SPLIT/SplitDecoder.py

Current source hashes:

```text
a03be9a05ae548478980db2b8c7fad627cf4fe159e5f009080e6416f72fda6c3  SplitDecoder.py
5c617885990850b5f8cc452efe34eeb500300c4a895737fcef2af38b5eaf3a3e  GiantTourEnv.py
90428e52037af9dbc3983e30a9042e094f23be1b514c1ed583463036e87412c6  GiantTourModel.py
3de7da676824cab5db80549b0e201bef6108015b069f4e2fa08cffa89640dfe6  GiantTourTrainer.py
c5fe7d68a420c9d495e3473642550aa100bd4d424491a4c66e063c0aca7e435c  train_n100.py
2570e02179e67a31a88628b229654943ef1de30713fb4e0048207a9c35b17789  test_pomo_split.py
```

Git-tree inspection also shows that all committed copies below are the exact
same blob `b2ff5105a4f6a1de5c1965dfc06ca26888e4b231`:

```text
POMO_SPLIT/SplitDecoder.py
POMO_SPLIT/result/20260711_212741.../src/SplitDecoder.py
POMO_SPLIT/result/20260711_213003.../src/SplitDecoder.py
POMO_SPLIT_EVALONLY/SplitDecoder.py
```

Thus the current two variants differ in reward wiring, not in their committed
Split implementation. An uncommitted CVRP100 run snapshot still needs its own
hash check.

## Independent exhaustive verification

The standalone verifier `work/audit_split_against_bruteforce.py` treats the
production `SplitDecoder.py` as the system under test. Its oracle independently
enumerates all `2^(n-1)` contiguous cut patterns and directly sums route costs.
It does not import the repository's test helper.

Configuration and result:

```text
source SHA256: a03be9a05ae548478980db2b8c7fad627cf4fe159e5f009080e6416f72fda6c3
seed: 20260713
n: 1..8
random cases per n: 30
batch: 3
tours per batch item: 5
random comparisons: 3600
absolute tolerance: 2e-5
maximum observed absolute error: 1.3352375827224705e-6
status: passed
```

Four explicit edge cases also passed: one full-capacity customer, exact capacity
boundary, all customers in one route, and every customer in a separate route.
Reconstructed routes preserved customer order/coverage and capacity.

## Exhaustive counterexamples for the modeling hypotheses

The pure-Python script `work/demonstrate_giant_tour_split_mismatch.py`
exhaustively enumerates every customer permutation and every feasible contiguous
partition for two fixed five-customer examples.

### Raw-tour objective versus post-Split objective

Even after choosing, among all raw-tour optima, the one with the best accidental
post-Split performance:

```text
best post-Split cost among raw-tour-optimal orders: 4.9570584433
globally best post-Split cost:                       4.3060433067
relative penalty:                                  15.1186388577%
```

Therefore a correct optimal Split cannot guarantee that a raw-tour-optimal
customer order is a good CVRP order.

### Same coordinates, different demands

With depot and all customer coordinates held exactly fixed, two demand vectors
were compared:

```text
number of optimal giant-tour encodings under demand A: 24
number of optimal giant-tour encodings under demand B: 24
common optimal encodings:                            0
best A-optimal order used under B:                  14.6917198319% worse
best B-optimal order used under A:                  17.6691554484% worse
```

This proves that demand-blind ordering can lose information needed for optimal
CVRP sequencing. It demonstrates possibility, not the magnitude expected on the
official CVRP100 distribution; that magnitude still requires a fixed-test-set
ablation.

## Official-POMO route recovery invariant

The standalone script `work/audit_split_recovers_pomo_routes.py` performs a
second independent structural check:

1. Run official POMO CVRP100 on the official fixed test set.
2. Remove every depot visit from each feasible POMO trajectory.
3. Preserve the resulting customer order as a giant tour.
4. Apply the candidate Split to that tour.

The original POMO route boundaries are one feasible contiguous partition, so an
optimal Split can never be more expensive. With official checkpoint SHA256
`d7a08f...d5ce8`, official test-set SHA256 `1bcf400f...ecc857`, and current-main
Split SHA256 `a03be9a...a6c3`, the CPU check produced:

```text
official CVRP100 instances: 10
POMO trajectories checked: 1000
tolerance: 2e-5
maximum Split minus original POMO cost: 4.76837158203125e-6
mean improvement from re-splitting: 0.0059991758
status: passed
```

This further supports correctness of the current Split formulation and its
customer-ID/depot/capacity conventions.

## Ready checkpoint-level demand diagnostic

`work/audit_giant_tour_demand_blindness.py` loads an exact giant-tour model
source and checkpoint, holds all coordinates fixed, cyclically reassigns each
instance's demand multiset, and performs two argmax rollouts. It then evaluates
both outputs with the selected Split source.

The harness was regression-tested with a checkpoint generated from current
`GiantTourModel.py`: 176 demand entries changed across two official CVRP100
instances, encoded tensors and all 20,000 selected tour elements remained
bit-for-bit identical, while all 200 post-Split trajectory costs changed. The
real poor checkpoint must still be tested; the dummy-checkpoint result validates
the harness, not the trained model's quality.

## Findings

### 1. No Split correctness defect is currently demonstrated

The current `SplitDecoder.py` matches the Prins/Jia shortest-path formulation,
and independent exhaustive tests found no disagreement. This is meaningful
evidence about commit `15338053`, but not yet about an unverified training
snapshot.

### 2. The exact reward variant is critical

`POMO_SPLIT` and `POMO_SPLIT_EVALONLY` are not equivalent experiments:

- `POMO_SPLIT` optimizes the same post-Split distance that evaluation reports.
- `POMO_SPLIT_EVALONLY` explicitly optimizes a different raw-tour objective and
  introduces capacity/depot route boundaries only during evaluation.

If the poor checkpoint is from `POMO_SPLIT_EVALONLY` or an equivalent
`POMO_TOUR` pipeline, the senior's explanation is a direct objective mismatch,
not merely a vague architectural suspicion.

### 3. Demand blindness is real in every inspected giant-tour model

For fixed depot/customer coordinates, changing only customer demands cannot
change model logits or the argmax giant tour. Yet those demands can change which
route boundaries and customer groupings are desirable. Split can optimize cuts
for the given order but cannot repair the order itself.

### 4. Depot information is not absent

The depot has a separate coordinate embedding and participates in encoder
context. It cannot be selected as an action. Any report should say "the policy
does not choose depot returns" rather than "the policy cannot see the depot."

### 5. Jia et al.'s mismatch is relevant but must be mapped carefully

The paper explains that giant-tour edges can be cut while depot-customer edges
appear only after Split. It reports stronger mismatch when many routes create
many depot edges or when depot edges form a large fraction of total distance.

This maps directly to raw-tour/eval-only training. For `POMO_SPLIT`, the
post-Split reward removes the explicit objective mismatch, although terminal
credit assignment and demand-blind inputs can still limit learning.

## Required next checks

1. Identify the exact poor checkpoint, module, result directory, run log, and
   copied source snapshot on the old server.
2. Hash/diff that snapshot against the current source.
3. Run the standalone exhaustive verifier against the snapshot's
   `SplitDecoder.py` on CPU.
4. Perform the POMO-route concatenation invariant: concatenate official POMO
   routes into a giant tour and confirm optimal Split is no worse than the
   original feasible partition.
5. Demonstrate demand blindness with the same coordinates and two demand
   assignments before proposing a demand-aware retraining ablation.

Until steps 1-3 are complete, the defensible conclusion is:

> Current main's Split appears correct. The leading explanations are either a
> raw-tour versus post-Split objective mismatch, demand-blind ordering, or a
> run/snapshot difference. The exact poor run has not yet been tied to one of
> these explanations.

The controlled follow-up ablations and their interpretation rules are specified
in `POMO_SPLIT_NEXT_EXPERIMENTS.md`.
