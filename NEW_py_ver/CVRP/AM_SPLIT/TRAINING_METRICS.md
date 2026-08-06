# Training metrics and paper plots

Both `train_am.py` and `train_am_split.py` use the official AM-scale defaults:

- 100 epochs
- 1,280,000 generated instances per epoch
- batch size 512 (2,500 optimizer steps per epoch)
- rollout baseline with one warmup epoch
- checkpoint every 10 completed epochs

Every training run writes one structured record file inside its timestamped output directory:

- `training_metrics.csv`: all metadata, batch metrics, epoch summaries, checkpoint events and
  the final run summary. Filter its `record_type` column by `run_metadata`, `batch`, `epoch`,
  `checkpoint`, `event` or `run_summary`. Full configuration and environment metadata are in
  the `details_json` cell of the `run_metadata` row; failure traceback is in the final row.
- `args.json` is retained only for compatibility with the official AM implementation.

Batch rows record every step by default. To reduce file size, pass
`--metrics-log-interval 10`; epoch statistics are still aggregated from every batch.

## Recommended paper figures

Compare two completed runs with:

```bash
python plot_training_metrics.py \
  AM=/path/to/am_run \
  AM-Split=/path/to/am_split_run \
  --output-dir /path/to/paper_plots
```

The command produces PNG and vector PDF versions of:

1. validation cost versus epoch, with standard-error bands;
2. validation cost versus cumulative wall-clock time (anytime efficiency);
3. train cost, REINFORCE loss and gradient norm diagnostics;
4. epoch time, throughput and peak GPU memory.

For the final article, also report mean and standard error across multiple independent
training seeds. The per-run validation standard error measures test-instance uncertainty;
it is not a replacement for between-seed variation.
