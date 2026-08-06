# POMO giant-tour + Split: next-experiment decision memo

## First principle

Do not change the model until the exact poor checkpoint is tied to its source
snapshot and reward mode. `POMO_SPLIT` and `POMO_SPLIT_EVALONLY` test different
hypotheses even though they share the same Split decoder and policy architecture.

## Critical causal point

Adding customer demand to the encoder is **not sufficient** when training still
uses the raw uncapacitated tour reward.

In `POMO_SPLIT_EVALONLY`:

```text
training reward = -(depot -> every customer once -> depot)
```

This reward is independent of demand and capacity. Even if demand is supplied as
an input, REINFORCE has no objective-level incentive to learn a demand-dependent
ordering. Demand can only help when the training signal itself changes with
demand, such as the post-Split CVRP cost or a justified demand-aware surrogate.

The original research goal can still be preserved:

- the policy selects customers only;
- depot remains masked and is never selected as an action;
- Split remains solely responsible for route boundaries/depot returns;
- the policy is trained with the scalar cost produced after Split.

## Minimal ablation matrix

Run these as distinct named experiments. Do not combine changes in one jump.

| ID | Policy inputs | Training reward | Purpose |
|---|---|---|---|
| `B0` | Official POMO CVRP inputs | Direct CVRP route cost | Published baseline |
| `G0` | Coordinates, depot context; no demand | Raw one-tour cost | Current eval-only hypothesis |
| `G1` | Coordinates, depot context; no demand | Post-Split CVRP cost | Isolate objective alignment |
| `G2` | Coordinates + customer demand + depot context | Post-Split CVRP cost | Isolate demand information |
| `G3` | `G2` plus explicit depot-relative geometry | Post-Split CVRP cost | Optional only if `G2` remains weak |

`G0` corresponds to `POMO_SPLIT_EVALONLY`; `G1` corresponds to current
`POMO_SPLIT`. `G2` should be the first new implementation after their exact runs
are verified.

## Smallest defensible G2 code change

Keep the action space and Split unchanged. Change only customer encoding:

```text
customer feature = [x, y, normalized_demand]
depot feature    = [x, y] through its existing separate depot projection
```

Required wiring:

1. `pre_forward` passes `reset_state.node_demand` to the encoder.
2. Customer embedding changes from `Linear(2, embedding_dim)` to
   `Linear(3, embedding_dim)`.
3. Demand is concatenated as `node_demand.unsqueeze(-1)`.
4. Training uses post-Split cost, not raw-tour cost.
5. Depot remains permanently masked as an action.
6. Split source and capacity convention remain unchanged.

Do not add "remaining capacity" as a dynamic decoder state in this first
ablation. A pure giant-tour policy has not yet chosen route boundaries, so its
remaining capacity is undefined until Split. Customer demand and fixed global
capacity are well-defined; route-state capacity is not.

Because CVRP100 capacity is always normalized to `1`, an explicit capacity scalar
cannot explain within-setting variation. It matters mainly if one model is meant
to generalize across capacities/scalers.

## Fair training protocol

For `G0`, `G1`, and `G2`, hold all of these fixed:

- official CVRP100 random generator and demand scaler;
- problem size and `pomo_size=100`;
- optimizer, learning rate, scheduler, train episodes per epoch, and total
  training budget;
- initialization policy and random seeds;
- checkpoint-selection rule;
- official fixed `vrp100_test_seed1234.pt` evaluation set;
- argmax decoding, no-augmentation and x8 settings;
- evaluator and Split implementation.

Use at least three training seeds before making a method-level claim. Report each
seed and mean/standard deviation; one checkpoint can be misleading.

## Diagnostics to record with every model

Besides final no-aug/x8 CVRP length, record:

- raw giant-tour length before Split;
- post-Split cost;
- number of decoded routes;
- total depot-boundary cost
  `sum(depot->first + last->depot)`;
- internal customer-customer cost retained after Split;
- customer-customer cost cut away by Split;
- runtime under the same timing protocol.

Bucket test instances by:

- estimated minimum route count `ceil(sum(demand)/capacity)`;
- mean/median customer distance to depot;
- total demand and demand variance.

If the Jia-style mismatch is the main cause, `G0`'s gap should grow in buckets
with more routes or a larger depot-edge contribution.

## Interpretation decision tree

1. **Split verifier fails:** fix Split/index/cost handling first; model conclusions
   are invalid.
2. **`G1` clearly beats `G0`:** raw-tour versus post-Split objective mismatch is
   causal.
3. **`G2` clearly beats `G1`:** missing demand information is causal.
4. **`G1` and `G2` both remain weak:** investigate terminal credit assignment,
   architecture, training convergence, POMO starts, and checkpoint selection.
5. **Only depot-distance-heavy buckets fail:** consider explicit relative-to-depot
   features or a boundary-aware auxiliary signal (`G3`).

## What can be told to the senior now

> Current-main Split matches the cited Bellman shortest-path formulation and has
> passed independent exhaustive and route-recovery checks. The main distinction
> is now between reward alignment and input information: raw-tour training does
> not optimize the edges introduced by Split, while all current giant-tour models
> are demand-blind. Adding demand alone is not enough if the reward still ignores
> demand. The clean next ablation is post-Split reward with demand-aware customer
> embeddings, keeping depot masked and Split unchanged.
