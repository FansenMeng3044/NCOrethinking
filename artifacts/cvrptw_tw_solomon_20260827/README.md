# CVRPTW TW training and Solomon evaluation artifacts

Created: 2026-08-27T12:00:48.501570+00:00

Source repository commit: `ad0abe30c9eb4d2f9b55cd79239b260b8e30d69c`

## Contents

- `checkpoints/`: eight final TW checkpoints only. POMO uses `checkpoint-2000.pt`; AM uses `epoch-100.pt`.
- `training_records/`: complete formal training logs, metrics, configuration, status, launch scripts, patches, and source snapshots from the two training batches, compressed with zstd.
- `solomon/solomon_tw_eval_20260827_034100.tar.zst`: Solomon-50/100 data, evaluator, logs, all 448 route JSON files, strict-protocol archive, and MVMoE final-checkpoint results.
- `solomon/solomon_results.csv` and `solomon/solomon_summary.csv`: directly accessible final MVMoE result tables.
- `models_portable.json`: checkpoint metadata using paths relative to this folder.
- `MANIFEST.csv`: file sizes and SHA-256 checksums.

Intermediate checkpoints, smoke-test directories, Python caches, and stale PID files are intentionally omitted. The eight final checkpoints are retained byte-for-byte and are listed with their original source paths and SHA-256 checksums.

The Solomon results use the MVMoE Table 7-style protocol: greedy, 8-fold geometric augmentation, POMO size=n, distance-only selection, and no enforcement of the parsed Solomon maximum fleet count. Do not mix its gaps with a vehicle-count-first SINTEF official comparison.

## Extract archives

```bash
cat training_records/cvrptw6_fixedlr_20260822_154648.tar.zst.part-* | tar --use-compress-program=unzstd -xf -
tar --use-compress-program=unzstd -xf training_records/tw4_official_resume_20260817_131657.tar.zst
tar --use-compress-program=unzstd -xf solomon/solomon_tw_eval_20260827_034100.tar.zst
```
