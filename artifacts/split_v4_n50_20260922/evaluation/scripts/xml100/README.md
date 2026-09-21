# CVRPLIB XML100 evaluation

This directory turns the public MVMoE benchmark idea into a strict, resumable
XML100 evaluation for the four local `NCOrethinking` checkpoint families:

- `pomo`: direct capacity-aware POMO;
- `pomo_split`: demand-blind POMO giant tour followed by exact Split;
- `am`: direct Attention Model;
- `am_split`: demand-blind AM giant tour followed by exact Split.

The concise issue-by-issue academic audit is in [`AUDIT.md`](AUDIT.md).

It does **not** load an MVMoE checkpoint.  The untouched upstream source snapshot
under `upstream/Routing-MVMoE` is retained only for protocol provenance.  Its
current pinned commit is `af29e5af0595f94f3ecc3bc46d72df1089a62682`.

## Official data

XML100 is the official CVRPLIB collection introduced by Queiroga, Sadykov,
Uchoa, and Vidal.  It contains 10,000 CVRP instances with 100 customers and
10,000 proven-optimal solutions.  The four digits in `XML100_ABCD_EF` encode
depot position, customer position, demand distribution, and average route size.

Authoritative sources:

- <https://galgos.inf.puc-rio.br/cvrplib/en/xml100>
- <https://openreview.net/pdf?id=yHiMXKN6nTl>
- official archive: <https://galgos.inf.puc-rio.br/cvrplib/uploads/files/xml100/XML.7z>

The official archive SHA-256 is:

```text
ef3814ee4c26e7f1d09cf33a4c0da564109de04ee393afa3b46c7eb33473b24b
```

The 10,000 released instances are final-test data only.  Do not use their
instances, optimal costs, optimal routes, or attribute-specific outcomes for
training, validation, hyperparameter selection, checkpoint selection, or model
selection.  The supplied generator may be used to create separate training and
validation instances, but the release notes warn that reproducing the original
files requires Python 2.7; Python 3 produces different instances.

Run the download/extraction/integrity check once:

```bash
python prepare_xml100.py
```

This checks the archive hash, file pairing, 378-group cardinalities, all 10,000
instance formats, and two independent released optimum sources:
`OptimalCosts.ods` and every `.sol` `Cost` line.  Rechecking an already
extracted release without network access is:

```bash
python prepare_xml100.py --no-download --no-extract
```

`py7zr` is needed only for extraction.

Important release audit: in the archive with the SHA-256 above, all 10,000
`.sol` declared costs exactly equal `OptimalCosts.ods`, but 91 released route
texts contain a duplicate or missing customer (a few also fail to reproduce the
declared cost).  The validator records the exact files and evidence in
`xml100_reference_route_anomalies.json`; it never edits them.  Consequently,
official optimum **values** are used for Gap, while released route texts are not
trusted as a feasibility oracle.  Every model route must pass the stricter
independent verifier without exception.

## Formal protocol

CVRPLIB makes the instances, CVRP constraints, EUC_2D objective, and proven
optima official; it does **not** prescribe a neural decoding budget.  The
following method-specific neural protocol follows the public MVMoE defaults and
must be named explicitly rather than called “the official XML100 decoder”:

- greedy/argmax decoding;
- 8-fold geometric augmentation;
- POMO size equals the customer count (`100`);
- no sampling, beam search, repair, local search, active search, or TTO;
- seeded deterministic PyTorch/CuDNN algorithms with a fixed cuBLAS workspace
  configuration; the manifest records CUDA visibility and library/hardware identity;
- fixed input coordinate divisor `1000`, matching the XML generator range and
  the public MVMoE CVRPLIB loader;
- demands divided by each instance's capacity.

Candidate selection and reporting are deliberately stricter than the public
MVMoE `Tester.py`:

1. `EDGE_WEIGHT_TYPE=EUC_2D` is parsed and required.
2. Every edge is scored as `floor(sqrt(dx^2 + dy^2) + 0.5)` before summation.
3. Direct models retain their depot-delimited routes.  Split models are decoded
   by an exact Bellman capacity Split under that same integer edge objective.
4. Direct decoder actions are scored as the explicit path
   `depot -> actions -> depot`.  They are never closed with a raw cyclic
   `roll`, which would incorrectly connect the last customer to the first when
   the decoder omits endpoint depot tokens.
5. Both MVMoE-style `no_aug` and 8-fold results are retained, including both
   complete route sets.  Candidate selection uses official integer EUC_2D cost;
   ties use deterministic candidate order.
6. The selected route is checked in the evaluator and then rechecked by a
   separate integer-arithmetic implementation for range, coverage, duplicates,
   capacity, depot closure, vehicle count, and exact cost.
7. The proven optimum is loaded only after candidate generation and selection;
   it is never included in model input or used to select among candidates.
   At runtime, the `.sol` Cost must still equal the hash-locked CSV exported
   from `OptimalCosts.ods`.
8. Gap is `100 * (cost - optimum) / optimum`.

Modern CVRP does not fix the number of routes.  Vehicle count is recorded, but
it is neither constrained nor part of the XML100 objective.

## Checkpoints

Copy `models.example.json` and replace all eight paths.  Relative paths are
resolved relative to the JSON file.  `expected_epoch` is enforced so an earlier
checkpoint cannot silently enter the comparison.  The formal launcher requires
an exact 64-hex `expected_sha256` for every checkpoint; compute and freeze these
before looking at any XML100 outcome.

The evaluator rejects TW checkpoints:

- POMO checkpoints must contain `model_state_dict` and strictly match the
  selected direct or Split architecture.
- AM checkpoints must contain `model`, have a sibling `args.json`, report a
  `graph_size` exactly matching the declared `training_size` (50 or 100), and
  declare problem `cvrp` or `am_split` as appropriate. XML100 itself always has
  100 customers, so n50 checkpoints are explicitly reported as n50-to-n100
  cross-size inference and are never presented as n100-trained models.

Every output records checkpoint path, epoch, size, top-level keys, and SHA-256.
The formal launcher also refuses a dirty source tree or a source branch/commit
other than the pinned values.  Override `EXPECTED_REPO_COMMIT` and
`EXPECTED_REPO_BRANCH` only when the new source revision has been deliberately
audited and reported.

## Run

Use a new output directory for a formal run:

```bash
bash run_xml100.sh \
  /root/autodl-tmp/NCOrethinking \
  /root/autodl-tmp/xml100_eval/models.json \
  /root/autodl-tmp/xml100_eval/results/final_checkpoints \
  cuda:0
```

For an interrupted run, invoke `evaluate_xml100.py` again with exactly the same
arguments plus `--resume`.  Each model-instance JSON, including its routes, is
written through a temporary file and atomic rename.  Resume verifies the run
manifest and checkpoint hashes before skipping existing results.  A non-resume
run refuses to overwrite a non-empty output directory.

A one-instance implementation smoke test is permitted, but is not a benchmark:

```bash
python evaluate_xml100.py \
  --repo /path/to/NCOrethinking \
  --models /path/to/models.json \
  --output /new/path/smoke \
  --device cuda:0 \
  --limit 1
```

## Outputs

- `run_manifest.json`: code/data/checkpoint/protocol/hardware identity;
- `solutions/<model>/<instance>.json`: selected routes and complete evidence;
- the same JSON contains both `routes` (8-fold) and `no_aug_routes`;
- `xml100_results.csv`: one canonical row per `(model, instance)`;
- `xml100_verification.json`: independent full-route verification report;
- `xml100_summary.csv`: overall, attribute-level, and all 378 group summaries;
- `xml100_overall.md`: compact paper-review table with success, augmented and
  no-augmentation Gap, vehicles, and timing;
- `xml100_pairwise.csv`: paired wins/ties/losses, gap differences, and, when
  SciPy is available, two-sided Wilcoxon signed-rank p-values with Holm
  correction;
- `xml100_pairwise_no_aug.csv`: the same paired analysis before augmentation;
- `xml100_summary_metadata.json`: metric definitions and interpretation notes.

Any individual saved solution can be visualized without rerunning inference:

```bash
python plot_xml100_solution.py \
  --instance data/XML/instances/XML100_1111_01.vrp \
  --solution-json results/final_checkpoints/solutions/pomo_n100/XML100_1111_01.json \
  --variant aug \
  --output figures/XML100_1111_01_pomo.png
```

Report success rate first.  When it is 100%, the primary quality aggregate is
the instance-weighted arithmetic mean optimality gap; otherwise all mean gaps
are explicitly conditional on successful instances and cannot outrank a fully
successful method by themselves.  The summary also reports median/tail gaps,
confidence interval, absolute error, optimal hits, vehicles, timing, and a macro
mean over complete XML100 groups.
Mean raw distance is descriptive only: it must not be used to rank methods over
heterogeneous groups.

## Comparability caveats

- POMO produces `8 x 100 = 800` candidates per instance, while greedy AM
  produces 8.  This is the standard algorithmic definition, but quality must be
  interpreted together with candidate count and wall time.
- Split is an explicit exact capacity postprocessor; Direct-vs-Split comparisons
  therefore compare complete algorithms, not only neural encoders.
- GPU and CPU algorithms are not directly time-comparable without reporting
  hardware, batching, and total preprocessing/postprocessing time.  The manifest
  and result rows retain these values.
- The greedy protocol has no algorithmic sampling and the launcher requests
  deterministic kernels.  Bitwise comparisons across different GPU models,
  CUDA/PyTorch builds, or source revisions are still not assumed.  If a randomized sampling protocol is
  studied separately, use multiple seeds, new output directories, average
  results per instance, and report the full sampling budget.  Do not mix those
  rows with this greedy protocol.
- Do not compare these integer `EUC_2D` results with scores produced by rounding
  a continuous route total once at the end.  For precision: the pinned MVMoE
  CVRP environment already rounds each edge when `loc_scaler=1000`; its outer
  `round(total*1000)` is redundant.  This toolkit replaces the float32 path
  with exact CPU integer arithmetic and adds saved-route verification rather
  than claiming that MVMoE used only total-level rounding.
- The public MVMoE code is protocol provenance, not a drop-in evaluator for
  these checkpoint formats.  This implementation intentionally changes its
  final candidate scoring is recomputed audibly with the official integer
  objective, while retaining greedy decoding, augmentation, and POMO budgets.
  Report the checkpoint architecture and exact scoring implementation when
  comparing against an unmodified MVMoE table.
- A `pomo_split`/`am_split` score includes exact Split postprocessing.  It must
  not be presented as neural-only inference, and its CPU postprocessing time
  must be included.
- Only predeclared final n=100 checkpoints belong in the formal comparison.
  XML100 outcomes must not be used to pick checkpoints, alter batch-independent
  decoding settings, or decide which model to report; doing so leaks test data.
- Failure rows are never dropped.  Success rate is reported first; paired tests
  use only common successful instances and record that paired sample size, so a
  method with failures cannot appear better merely through a reduced subset.
- `no_aug` and 8-fold have different inference budgets.  Likewise POMO and AM
  have different candidate counts.  Quality tables must show candidate budget
  and time alongside Gap.
