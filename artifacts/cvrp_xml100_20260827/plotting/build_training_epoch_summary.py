#!/usr/bin/env python3
"""Export compact epoch-level plotting data from the eight full metrics CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "model",
    "architecture",
    "training_size",
    "epoch",
    "timestamp_utc",
    "elapsed_seconds",
    "cumulative_elapsed_seconds",
    "learning_rate_start",
    "learning_rate_end",
    "train_score_mean",
    "train_score_std",
    "train_cost_mean",
    "train_cost_std",
    "train_loss_mean",
    "train_loss_std",
    "reinforce_loss_mean",
    "baseline_loss_mean",
    "total_loss_mean",
    "grad_norm_mean",
    "grad_norm_max",
    "training_seconds",
    "checkpoint_seconds",
    "validation_seconds",
    "epoch_total_seconds",
    "validation_count",
    "validation_cost_mean",
    "validation_cost_std",
    "best_validation_cost_mean",
    "best_validation_epoch",
    "checkpoint_saved",
    "checkpoint_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def identify(path: Path) -> tuple[str, str, int]:
    normalized = path.as_posix().lower()
    cases = [
        ("pomo_split_cvrp100", "pomo_split_n100", "pomo_split", 100),
        ("pomo_split_cvrp50", "pomo_split_n50", "pomo_split", 50),
        ("pomo_cvrp100", "pomo_n100", "pomo", 100),
        ("pomo_cvrp50", "pomo_n50", "pomo", 50),
        ("am_split_100", "am_split_n100", "am_split", 100),
        ("am_split_50", "am_split_n50", "am_split", 50),
        ("/cvrp_100/", "am_n100", "am", 100),
        ("/cvrp_50/", "am_n50", "am", 50),
    ]
    for token, name, architecture, size in cases:
        if token in normalized:
            return name, architecture, size
    raise ValueError(f"cannot identify model from {path}")


def main() -> None:
    args = parse_args()
    inputs = sorted(args.metrics_root.rglob("training_metrics.csv"))
    if len(inputs) != 8:
        raise ValueError(f"expected 8 training_metrics.csv files, found {len(inputs)}")
    rows: list[dict[str, str | int]] = []
    counts: dict[str, int] = {}
    for path in inputs:
        model, architecture, size = identify(path)
        count = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                if source.get("record_type") != "epoch" or source.get("event") != "epoch_finished":
                    continue
                target: dict[str, str | int] = {
                    "model": model,
                    "architecture": architecture,
                    "training_size": size,
                }
                for field in FIELDS[3:]:
                    target[field] = source.get(field, "")
                rows.append(target)
                count += 1
        expected = 2000 if architecture.startswith("pomo") else 100
        if count != expected:
            raise ValueError(f"{model}: expected {expected} epoch rows, found {count}")
        counts[model] = count
    rows.sort(key=lambda row: (str(row["model"]), int(str(row["epoch"]))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"wrote {len(rows)} epoch rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
