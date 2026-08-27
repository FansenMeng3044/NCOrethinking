#!/usr/bin/env python3
"""Plot the eight final CVRP runs from the compact epoch-level summary."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


COLORS = {"n50": "#4C78A8", "n100": "#E45756"}
STYLES = {"direct": "-", "split": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def value(row: dict[str, str], field: str) -> float | None:
    text = row.get(field, "").strip()
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def label(model: str) -> str:
    return model.replace("pomo", "POMO").replace("am", "AM").replace("_split", "-Split").replace("_n", " n")


def draw(axis, grouped, models, field, title, ylabel):
    for model in models:
        points = [
            (int(row["epoch"]), metric)
            for row in grouped[model]
            if (metric := value(row, field)) is not None
        ]
        if not points:
            continue
        size = "n100" if model.endswith("n100") else "n50"
        style = STYLES["split" if "split" in model else "direct"]
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=COLORS[size],
            linestyle=style,
            linewidth=1.45,
            label=label(model),
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, fontsize=9)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    draw(
        axes[0, 0],
        grouped,
        ["pomo_n50", "pomo_split_n50", "pomo_n100", "pomo_split_n100"],
        "train_score_mean",
        "POMO training score",
        "Reported mean route cost (lower is better)",
    )
    draw(
        axes[0, 1],
        grouped,
        ["pomo_n50", "pomo_split_n50", "pomo_n100", "pomo_split_n100"],
        "train_loss_mean",
        "POMO policy loss",
        "Mean loss",
    )
    draw(
        axes[1, 0],
        grouped,
        ["am_n50", "am_split_n50", "am_n100", "am_split_n100"],
        "train_cost_mean",
        "AM training cost",
        "Mean cost (lower is better)",
    )
    draw(
        axes[1, 1],
        grouped,
        ["am_n50", "am_split_n50", "am_n100", "am_split_n100"],
        "validation_cost_mean",
        "AM fixed-set validation cost",
        "Mean cost (lower is better)",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in [item.strip() for item in args.formats.split(",") if item.strip()]:
        figure.savefig(args.output.with_suffix(f".{extension}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"plotted {len(rows)} epoch rows from {args.summary}")


if __name__ == "__main__":
    main()
