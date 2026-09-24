# CVRPLIB X-set evaluation

This artifact contains the complete CVRPLIB X-set evaluation of the final
100-customer AM Direct, AM Split, POMO Direct, and POMO Split models. It
includes the benchmark instances and reference solutions, the exact evaluation
and verification code, all 400 returned solutions, per-instance and aggregate
results, the complete execution log, and the appendix tables.

## Protocol

- Benchmark: all 100 CVRPLIB X instances, ranging from 100 to 1,000 customers.
- Training size: 100 customers for every evaluated model.
- AM inference: greedy decoding with eightfold geometric augmentation.
- POMO inference: one starting point per customer with eightfold geometric
  augmentation.
- Candidate selection and reporting: total distance under the official integer
  `EUC_2D` convention.
- Direct models: no route postprocessing.
- Split models: exact capacity-feasible Split decoding after customer-order
  construction.
- Reference `.sol` files: read only after candidate generation and selection.
- Feasibility: every returned solution is independently checked for customer
  coverage, duplicate visits, vehicle capacity, route structure, and objective
  value.

The complete run used a single NVIDIA RTX 6000D with PyTorch 2.8.0 and CUDA
12.8. Deterministic PyTorch and cuDNN settings were enabled. The measured wall
time was 1,000.88 seconds.

## Aggregate results

| Method | Mean gap to reference | Median gap | Mean vehicles | Feasible |
|:--|--:|--:|--:|--:|
| AM Direct | 13.540% | 11.623% | 56.20 | 100/100 |
| AM Split | 19.126% | 17.166% | 56.72 | 100/100 |
| POMO Direct | 10.370% | 8.940% | 54.23 | 100/100 |
| POMO Split | 19.005% | 17.945% | 56.18 | 100/100 |

In the paired comparison, AM Direct obtains a lower objective value on 90 of
the 100 instances and AM Split on 10. POMO Direct obtains a lower objective
value on 91 instances and POMO Split on 9. There are no ties. The average gap
between Split and Direct is 5.586 percentage points for AM and 8.635 percentage
points for POMO.

The values in this artifact use the costs reported in the CVRPLIB solution
files as reference values. They are not relabeled as proven optima.

## Contents

- `data/X/`: the 100 `.vrp` instances and corresponding `.sol` files.
- `evaluation/`: the executed evaluator, parser, summarizer, original model
  configuration, and repository-relative model configuration.
- `results/xset_results.csv`: complete evaluator output before the independent
  replay export.
- `results/xset_verified_results.csv`: compact per-instance results after
  independent verification.
- `results/xset_summary.json`: aggregate statistics.
- `results/run_manifest.json`: hardware, software, checkpoint, protocol,
  determinism, timing, and source provenance.
- `results/solutions/`: all 400 route-level JSON solution files.
- `logs/xset_full.log`: complete formal-run log.
- `appendix/`: centered aggregate, paired, and per-instance LaTeX tables,
  together with their generator and audit record.

`evaluation/models_xset.json` is the unchanged configuration used on the
evaluation server and is retained for provenance. For a fresh repository clone,
use `evaluation/models_xset_repo.json`, whose checkpoint paths are relative to
this artifact.

## Checkpoints

| Model | Repository path | SHA-256 |
|:--|:--|:--|
| AM Direct | `artifacts/cvrp_xml100_20260827/checkpoints/am_n100/epoch-100.pt` | `450c46de4a257605d66b5705cb38f0fd265ec3587a2c7bd866d0605ee463dfce` |
| AM Split | `artifacts/split_v4_n100_20260922/checkpoints/am_split_cvrp100/epoch-100.pt` | `58cdc39a2132e475848f48a0908ba3afe0e07720fb59fb6b50809a1e2b53170b` |
| POMO Direct | `artifacts/cvrp_xml100_20260827/checkpoints/pomo_n100/checkpoint-2000.pt` | `74bf36630f179b69d69ceae4997d6b03994815677c0c6bf4000b4936feb29af8` |
| POMO Split | `artifacts/split_v4_n100_20260922/checkpoints/pomo_split_cvrp100/checkpoint-2000.pt` | `4333ab8812c0443e723d3d1b4f1e39b56f4f1045dc435889e03403aa14f4eb8a` |

## Reproduction

Run from the repository root. Use a new output directory rather than
overwriting the recorded formal results.

```bash
python artifacts/cvrplib_x_eval_20260924/evaluation/evaluate_xset.py \
  --repo . \
  --models artifacts/cvrplib_x_eval_20260924/evaluation/models_xset_repo.json \
  --data artifacts/cvrplib_x_eval_20260924/data/X \
  --output artifacts/cvrplib_x_eval_20260924/reproduction_output \
  --device cuda:0 \
  --augmentation 8 \
  --require-checkpoint-hashes

python artifacts/cvrplib_x_eval_20260924/evaluation/summarize_xset.py \
  --data artifacts/cvrplib_x_eval_20260924/data/X \
  --output artifacts/cvrplib_x_eval_20260924/reproduction_output

python artifacts/cvrplib_x_eval_20260924/appendix/build_xset_appendix.py
```

The formal-run manifest records SHA-256 hashes for the evaluator sources,
model configuration, dataset identity, and every checkpoint. `CHECKSUMS.sha256`
adds hashes for the principal result and appendix files in this repository.
