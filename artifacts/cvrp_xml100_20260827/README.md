# Final CVRP checkpoints, training records, and XML100 evaluation

This self-contained archive preserves the final eight non-time-window CVRP
checkpoints, their training evidence, the official CVRPLIB XML100 release, the
audited evaluation toolkit, and plotting-ready formal results.

It is separate from `../cvrptw_tw_solomon_20260827/`; no TW/Solomon file was
overwritten.

## Contents

- `checkpoints/`: POMO, POMO-Split, AM, and AM-Split at n50 and n100.
  The four POMO files are `checkpoint-2000.pt`; the four AM files are
  `epoch-100.pt`. AM `args.json` and POMO `job_config.json` are colocated.
- `server_metadata/checkpoint_validation.json`: CPU load verification, required
  keys, byte sizes, epochs, SHA-256 values, and optimizer learning rates. All
  eight recorded learning rates are exactly `0.0001`.
- `training_records/`: metrics CSVs, logs, arguments/configuration, scripts,
  evaluation logs, and status from the two original experiment directories.
  Intermediate `.pt` files are deliberately excluded because the eight final
  checkpoints are stored once, unmodified, under `checkpoints/`. A compact
  direct-use `training_metrics_epoch_summary.csv` contains all 8,400 completed
  epoch rows, so training curves do not require unpacking the large CSVs.
  `training_final_epoch_metrics.csv` and the adjacent validation JSON provide
  an eight-row handoff and confirm every epoch-level learning rate is `0.0001`.
- `xml100/dataset/XML.7z`: the official 10,000-instance/10,000-solution XML100
  release. Its SHA-256 is
  `ef3814ee4c26e7f1d09cf33a4c0da564109de04ee393afa3b46c7eb33473b24b`.
- `xml100/code/`: preparation, inference, exact Split, independent route
  verification, summary, route plotting, and protocol tests. The untouched
  public MVMoE `Tester.py` snapshot is included for protocol provenance only.
- `xml100/results/`: canonical 40,000-row CSV and paper-ready summaries for the
  four n100 checkpoints. Every one of 10,000 instances succeeded for every
  model and every saved route was independently verified.
- `xml100/routes/xml100_n100_final_20260827.tar.zst`: the full evaluator output,
  including all 40,000 per-model/per-instance route JSON files. Each JSON keeps
  both the no-augmentation route and the selected 8-fold route.
- `plotting/`: single-route and aggregate plotting tools plus pre-generated
  PNG/PDF figures.

`models_portable.json` is the hash-locked eight-checkpoint configuration. The
formal 40,000-row run used its four n100 entries; the n50 files are archived for
cross-size experiments but are not mixed into the reported n100 table.

## Formal XML100 protocol

The run used greedy decoding, 8-fold geometric augmentation, POMO size 100, no
sampling/beam/repair/local search/TTO, official per-edge integer `EUC_2D`, exact
capacity Split for Split models, and Gap against the released proven optimum.
Vehicle count is descriptive only and is not an XML100 objective.

Overall augmented mean Gaps were: POMO 5.9040%, POMO-Split 8.0076%, AM 8.7273%,
and AM-Split 12.1613%. Read `xml100/results/xml100_overall.md` and the canonical
CSV before drawing conclusions; the complete attribute/group and pairwise
statistics are also included.

## Reconstruct training records

The larger training archive is split below GitHub's single-file limit. This
portable command reconstructs both logical `.tar.zst` files and verifies every
part and reconstructed SHA-256:

```bash
python training_records/reassemble_training_archives.py --output-dir extracted/training_archives
```

Extract with modern GNU tar and zstd, for example:

```bash
tar --use-compress-program=unzstd -xf extracted/training_archives/eight_models_cvrp50_100_20260806_230300.tar.zst
```

The exact archive membership is listed in the two `*.contents.txt` files.

## Extract data and routes

Prepare and re-audit the official data without downloading anything:

```bash
python xml100/code/prepare_xml100.py \
  --archive xml100/dataset/XML.7z \
  --data-root extracted/xml100_data \
  --manifest extracted/xml100_dataset_manifest.json \
  --no-download
```

Extract all route JSON files:

```bash
tar --use-compress-program=unzstd \
  -xf xml100/routes/xml100_n100_final_20260827.tar.zst \
  -C extracted
```

## Plot without rerunning inference

Generate aggregate PNG and PDF figures directly from the canonical CSV:

```bash
python plotting/plot_xml100_aggregate.py \
  --results xml100/results/xml100_results.csv \
  --output-dir figures
```

Replot all eight training runs from the compact epoch table:

```bash
python plotting/plot_training_curves.py \
  --summary training_records/training_metrics_epoch_summary.csv \
  --output figures/training_curves
```

Visualize one saved route after extracting the dataset and route archive:

```bash
python plotting/plot_xml100_solution.py \
  --instance extracted/xml100_data/XML/instances/XML100_1111_01.vrp \
  --solution-json extracted/xml100_n100_final_20260827/solutions/pomo_n100/XML100_1111_01.json \
  --variant aug \
  --output figures/XML100_1111_01_pomo.png
```

Use `MANIFEST.csv` for machine-readable file inventory and `SHA256SUMS` for
whole-archive integrity verification.
