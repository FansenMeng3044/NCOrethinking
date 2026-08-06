#!/usr/bin/env python
"""Create paper-ready AM/AM-Split training plots from epoch_metrics.csv files."""

import argparse
import csv
from pathlib import Path


def parse_run(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).name
    run_dir = Path(path).expanduser().resolve()
    metrics_path = run_dir / "training_metrics.csv" if run_dir.is_dir() else run_dir
    return label, metrics_path


def read_metrics(path):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("record_type") == "epoch"]
    if not rows:
        raise ValueError("No epoch rows found in {}".format(path))
    return rows


def values(rows, key):
    result = []
    for row in rows:
        value = row.get(key, "")
        result.append(float(value) if value not in ("", None) else float("nan"))
    return result


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / "{}.{}".format(stem, suffix), dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", metavar="LABEL=RUN_DIR",
        help="One or more run directories, optionally prefixed with a legend label",
    )
    parser.add_argument("--output-dir", default="paper_plots")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [(label, read_metrics(path)) for label, path in map(parse_run, args.runs)]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, rows in runs:
        epoch = values(rows, "epoch")
        mean = values(rows, "validation_cost_mean")
        se = values(rows, "validation_standard_error")
        line = ax.plot(epoch, mean, marker="o", markersize=3, label=label)[0]
        ax.fill_between(epoch, [m - e for m, e in zip(mean, se)],
                        [m + e for m, e in zip(mean, se)], alpha=0.18, color=line.get_color())
    ax.set(xlabel="Epoch", ylabel="Greedy validation cost", title=args.title)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, "validation_cost_vs_epoch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, rows in runs:
        hours = [v / 3600.0 for v in values(rows, "cumulative_elapsed_seconds")]
        ax.plot(hours, values(rows, "validation_cost_mean"), marker="o", markersize=3, label=label)
    ax.set(xlabel="Cumulative wall-clock time (hours)", ylabel="Greedy validation cost", title=args.title)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, "validation_cost_vs_time")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    for label, rows in runs:
        epoch = values(rows, "epoch")
        axes[0].plot(epoch, values(rows, "train_cost_mean"), label=label)
        axes[1].plot(epoch, values(rows, "reinforce_loss_mean"), label=label)
        axes[2].plot(epoch, values(rows, "grad_norm_mean"), label=label)
    axes[0].set_ylabel("Train cost")
    axes[1].set_ylabel("REINFORCE loss")
    axes[2].set_ylabel("Gradient norm")
    axes[2].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    if args.title:
        fig.suptitle(args.title)
    save_figure(fig, output_dir, "training_diagnostics")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    for label, rows in runs:
        epoch = values(rows, "epoch")
        axes[0].plot(epoch, values(rows, "epoch_total_seconds"), label=label)
        axes[1].plot(epoch, values(rows, "throughput_instances_per_second"), label=label)
        axes[2].plot(epoch, values(rows, "gpu_peak_allocated_mb"), label=label)
    axes[0].set_ylabel("Epoch wall time (s)")
    axes[1].set_ylabel("Instances / s")
    axes[2].set_ylabel("Peak GPU memory (MiB)")
    axes[2].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    if args.title:
        fig.suptitle(args.title)
    save_figure(fig, output_dir, "training_efficiency")
    plt.close(fig)

    print("Saved plots to {}".format(output_dir))


if __name__ == "__main__":
    main()
