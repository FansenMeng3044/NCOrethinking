# POMO giant-tour + Split audit handoff

You are the field agent on the old 2x RTX 4090 server. Work carefully and report
evidence to the main Codex conversation. Do not edit source code, stop jobs, move
result directories, or start retraining/evaluation in this phase.

## Context

The repository now contains more than one giant-tour experiment. All variants
change POMO CVRP100 from direct route construction to:

1. The policy outputs a permutation of all customers (a giant tour).
2. A capacity-constrained Split decoder partitions that fixed order into routes.
3. A training reward is assigned to the permutation. This is the crucial point
   on which the variants differ and must not be inferred from directory names.

As of remote `main` commit `15338053`, distinguish at least:

- `POMO_SPLIT`: training and evaluation both use negative post-Split CVRP cost.
- `POMO_SPLIT_EVALONLY`: training minimizes one uncapacitated
  `depot -> all customers -> depot` tour; Split is used only for evaluation.
- `POMO_TOUR`: another raw-tour environment/model; determine how the senior's
  run actually wires it to training and evaluation.

The senior's explanation that the policy learns a short customer tour but depot
boundary edges become long after Split sounds most like the eval-only/raw-tour
variant. Treat that as a hypothesis, not proof of which checkpoint was run.

The effect is currently reported as poor. We must first determine whether this is
an implementation/run-provenance problem or an expected limitation of the idea.

The reference named by the senior is:

- Jia, Mei, Zhang (2022), *Confidence-Based Ant Colony Optimization for
  Capacitated Electric Vehicle Routing Problem With Comparison of Different
  Encoding Schemes*.
- Its indirect encoder calls `split(T, Qc)` and explicitly cites Prins (2004),
  *A Simple and Effective Evolutionary Algorithm for the Vehicle Routing
  Problem*, as reference [29].

The classical specification to audit against is:

- Given a customer permutation `pi[0:n]`, create auxiliary vertices `0..n`.
- An arc `(i,j)`, `i < j`, represents exactly one route
  `depot -> pi[i] -> ... -> pi[j-1] -> depot`.
- The arc exists iff the segment demand is at most capacity `Q` (CVRP100 has no
  maximum-route-length constraint).
- Its weight includes both depot boundary edges and every internal edge exactly
  once.
- A shortest path from auxiliary vertex `0` to `n` gives the minimum-total-length
  capacity-feasible contiguous partition of that fixed permutation.
- Split optimizes only the partition. It never reorders customers.

## Phase A: identify the exact run

Start in the training repository, expected at `/data/data2/mfs/pomo`:

```bash
cd /data/data2/mfs/pomo
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline

find NEW_py_ver/CVRP -maxdepth 4 -type f \
  \( -name 'run_log.txt' -o -name 'checkpoint-*.pt' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
  | sort
```

Identify the exact CVRP100 result/checkpoint that the senior called "poor". For
that run, record:

- experiment module (`POMO_SPLIT`, `POMO_SPLIT_EVALONLY`, `POMO_TOUR`, or
  another local variant), result directory, and checkpoint epoch/path;
- checkpoint SHA256;
- `run_log.txt` settings and final training score/loss;
- Git commit and dirty state at launch if recorded;
- hashes/diffs between the run's `src/` snapshot and the current files below.

Do not assume current source equals training source. The run's copied `src/` is
the primary evidence.

```bash
sha256sum \
  NEW_py_ver/CVRP/POMO_SPLIT/SplitDecoder.py \
  NEW_py_ver/CVRP/POMO_SPLIT/GiantTourEnv.py \
  NEW_py_ver/CVRP/POMO_SPLIT/GiantTourModel.py \
  NEW_py_ver/CVRP/POMO_SPLIT/GiantTourTrainer.py \
  NEW_py_ver/CVRP/POMO_SPLIT/train_n100.py

find NEW_py_ver/CVRP/POMO_SPLIT_EVALONLY NEW_py_ver/CVRP/POMO_TOUR \
  -maxdepth 2 -type f \( -name '*.py' -o -name 'README.md' \) -print0 \
  | sort -z | xargs -0 sha256sum
```

## Phase B: line-by-line implementation audit

Read the exact run snapshot and current source. Report file/line references for
every conclusion.

### Split decoder invariants

Verify all of these:

1. Every rollout is a permutation of customer IDs `1..n`, with no depot,
   duplicate, or missing customer.
2. Customer IDs are mapped to coordinates and demands without an off-by-one.
3. Segment load is the sum for precisely `pi[i:j]` and feasibility is
   `load <= Q + epsilon`.
4. Segment cost is
   `d(depot,pi[i]) + sum(d(pi[k],pi[k+1])) + d(pi[j-1],depot)`.
5. The dynamic program considers every feasible `i < j`; it is not greedy
   capacity filling.
6. The label recurrence is `V[j] = min_i(V[i] + segment_cost(i,j))` with
   `V[0]=0`.
7. Predecessor reconstruction covers each customer once and produces only
   capacity-feasible routes.
8. The returned objective is the sum of all route lengths, not giant-tour
   length, route count, or makespan.
9. No fixed vehicle-count or route-length constraint has been introduced. Those
   constraints are absent from the official POMO CVRP100 setting.
10. Coordinates use the same Euclidean distance and demands use the same
    normalized capacity convention as official `CVRProblemDef.py`.

### Policy/reward invariants

Verify separately:

1. Whether depot coordinates are model inputs or truly omitted.
2. Whether customer demands and capacity are model inputs.
3. Whether depot can ever be selected as an action.
4. The exact training reward: post-Split CVRP cost, depot-anchored raw tour, or
   customer-only TSP cycle.
5. The exact evaluation score and whether it differs from the training reward.
6. Whether REINFORCE receives the identified training reward while Split itself
   remains a discrete, no-gradient decoder.
7. Whether `pomo_size=100`, first-customer starts, sampling/argmax settings, and
   the actual training schedule match the run log.

Expected from remote `main` at commit `15338053` (verify the exact run snapshot,
do not merely repeat):

- depot coordinates are encoded as context;
- demand and capacity are not policy inputs;
- depot is permanently masked as an action;
- `POMO_SPLIT` trains on negative post-Split CVRP distance;
- `POMO_SPLIT_EVALONLY` trains on raw depot-anchored tour distance and switches
  to post-Split distance only in `GiantTourTester`;
- `POMO_TOUR` supports raw `single_route` and `tsp_cycle` rewards.

## Phase C: CPU correctness checks

These do not need a GPU. If the exact run used `POMO_SPLIT` or copied its unit
test, run the existing test from the exact source context:

```bash
cd /data/data2/mfs/pomo/NEW_py_ver/CVRP/POMO_SPLIT
python test_pomo_split.py
```

Then independently validate Split, without importing its test helper as the
oracle:

- generate deterministic random cases for `n=1..8`;
- enumerate all `2^(n-1)` contiguous cut patterns;
- reject capacity-infeasible partitions;
- directly sum depot-out, internal, and depot-return distances;
- compare the minimum against `split_giant_tours` within a stated tolerance;
- check exact-capacity, one-customer, one-route, and many-route edge cases;
- reconstruct routes and verify customer coverage and capacity.

The main conversation provides a ready independent verifier named
`audit_split_against_bruteforce.py`. Copy it outside the repository or put it in
`/tmp`, then run it once against current source and once against the exact run
snapshot, for example:

```bash
python /tmp/audit_split_against_bruteforce.py \
  --pomo-root /data/data2/mfs/pomo \
  --split-source /path/to/the/exact/run/src/SplitDecoder.py \
  --output /tmp/split_audit_exact_run.json
```

Put any temporary verifier under `/tmp`, not in the repository. Report the seed,
number of cases, tolerance, command, and complete outcome. A passing self-written
unit test alone is not enough; the oracle must be independent.

## Phase D: strongest consistency check

Run this small CPU check; it is not formal evaluation:

1. On a small fixed CVRP batch, obtain routes from official POMO.
2. Remove depot repeats and concatenate those routes into one giant customer
   sequence while preserving each route's internal order.
3. Apply the candidate Split to that sequence.
4. The Split cost must be no greater than the original POMO route cost, because
   the original route boundaries are one feasible contiguous partition.

If this inequality fails beyond floating-point tolerance, locate the exact cause
before discussing model quality.

The main conversation provides `audit_split_recovers_pomo_routes.py`:

```bash
python /tmp/audit_split_recovers_pomo_routes.py \
  --pomo-root /data/data2/mfs/pomo \
  --checkpoint /data/data2/mfs/pomo/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt \
  --test-data /data/data2/mfs/pomo/NEW_py_ver/CVRP/vrp100_test_seed1234.pt \
  --split-source /path/to/the/exact/run/src/SplitDecoder.py \
  --episodes 10 \
  --device cpu \
  --output /tmp/split_recovers_pomo_routes.json
```

For reference, current-main Split SHA256
`a03be9a05ae548478980db2b8c7fad627cf4fe159e5f009080e6416f72fda6c3`
passed this invariant locally on 1,000 trajectories from 10 official CVRP100
instances. The maximum `split - original` was `4.76837158203125e-6` under a
`2e-5` tolerance.

## Phase E: demand-blind checkpoint check

The main conversation also provides `audit_giant_tour_demand_blindness.py`. Run
it against the exact poor checkpoint and its copied model source. It keeps depot
and customer coordinates fixed, reassigns the same demand multiset among
customers, rolls out argmax twice, and scores both tours with Split.

For a `GiantTourModel` run:

```bash
python /tmp/audit_giant_tour_demand_blindness.py \
  --model-source /path/to/the/exact/run/src/GiantTourModel.py \
  --model-class GiantTourModel \
  --split-source /path/to/the/exact/run/src/SplitDecoder.py \
  --checkpoint /path/to/the/poor/checkpoint-N.pt \
  --test-data /data/data2/mfs/pomo/NEW_py_ver/CVRP/vrp100_test_seed1234.pt \
  --episodes 2 \
  --pomo-size 100 \
  --device cpu \
  --output /tmp/demand_blindness_exact_checkpoint.json
```

For a `POMO_TOUR` run, use its exact `TourModel.py` and
`--model-class TourModel`, while still pointing `--split-source` to the Split
implementation used at evaluation.

## Representation diagnosis

If Split and run provenance pass, explain these hypotheses without changing code:

- **Demand blindness:** with identical depot/customer coordinates but different
  demand vectors, an argmax policy with no demand input must produce identical
  customer orders, although the best capacity-aware order may change.
- **Split cannot repair ordering:** Split finds the best cuts for a fixed order;
  it cannot swap customers across or within routes.
- **Objective mismatch in eval-only/raw-tour training:** the policy is explicitly
  rewarded for customer edges that Split may later cut, while depot boundary edges
  introduced by Split do not appear as such in the training objective. This is the
  closest analogue of Jia et al.'s ACO pheromone mismatch.
- **Credit-assignment limitation in post-Split training:** `POMO_SPLIT` removes the
  objective mismatch by using post-Split cost, but still supplies only a terminal
  scalar reward for a discontinuous partitioning operation.
- **Conditions that amplify the issue:** large demands create many short routes
  and therefore many depot boundary edges; far-away depot geometry makes those
  edges a larger fraction of total cost.
- **Training fairness:** compare actual epoch, training instances, LR schedule,
  checkpoint selection, and convergence with the baseline before attributing all
  loss to architecture.

Propose, but do not run, these diagnostic ablations:

1. Same coordinates, two demand assignments: compare encoded nodes/tours and
   Split cost (Phase E provides the executable check).
2. Same fixed test set: compare giant tours from the new model, random order,
   nearest-neighbor order, and customer sequences derived from official POMO.
3. Bucket the performance gap by estimated route count
   `ceil(sum(demand)/capacity)` and a depot-distance statistic.
4. Compare coordinates-only against a demand-aware encoder while keeping Split,
   reward, training budget, checkpoint selection, and evaluator fixed.

## Required report format

Return findings in this order:

1. Exact run/checkpoint provenance.
2. Any correctness findings, highest severity first, with file/line references.
3. A table mapping each Prins/Jia requirement to the run's implementation.
4. CPU test commands and results.
5. POMO-route recovery invariant and demand-blind checkpoint-check results.
6. Whether the current evidence supports "Split bug", "wrong run/version",
   "raw-tour versus Split objective mismatch", "demand-blind representation",
   another training limitation, or remains inconclusive.
7. The smallest next diagnostic experiment.

Do not modify or commit anything in this audit. Do not launch training or formal
10k evaluation. Stop after the evidence report so the main conversation can
review it.
