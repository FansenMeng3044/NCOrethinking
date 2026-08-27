#!/usr/bin/env python3
"""Create plotting-ready XML100 benchmark figures from the canonical result CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


MODEL_ORDER = ["pomo_n100", "pomo_split_n100", "am_n100", "am_split_n100"]
MODEL_LABELS = {
    "pomo_n100": "POMO",
    "pomo_split_n100": "POMO-Split",
    "am_n100": "AM",
    "am_split_n100": "AM-Split",
}
MODEL_COLORS = {
    "pomo_n100": "#4C78A8",
    "pomo_split_n100": "#72B7B2",
    "am_n100": "#F58518",
    "am_split_n100": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="xml100_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated matplotlib formats (default: png,pdf)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def finite_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    models = [model for model in MODEL_ORDER if any(row.get("model") == model for row in rows)]
    extras = sorted({row.get("model", "") for row in rows if row.get("model") not in models})
    models.extend(model for model in extras if model)
    if not rows or not models:
        raise ValueError(f"no model rows found in {path}")
    return rows, models


def save_figure(figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    for extension in formats:
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")


def mean_for(rows: list[dict[str, str]], model: str, key: str, status_key: str = "status") -> float:
    values = [
        value
        for row in rows
        if row.get("model") == model and row.get(status_key) == "ok"
        if (value := finite_float(row, key)) is not None
    ]
    if not values:
        raise ValueError(f"no finite {key} values for {model}")
    return fmean(values)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, models = load_rows(args.results)
    labels = [MODEL_LABELS.get(model, model) for model in models]
    colors = [MODEL_COLORS.get(model, "#777777") for model in models]

    # Overall augmented vs. non-augmented optimality gap.
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    x = list(range(len(models)))
    width = 0.36
    aug = [mean_for(rows, model, "gap_percent") for model in models]
    no_aug = [mean_for(rows, model, "no_aug_gap_percent", "no_aug_status") for model in models]
    axis.bar([value - width / 2 for value in x], aug, width, label="8-fold", color=colors)
    axis.bar(
        [value + width / 2 for value in x],
        no_aug,
        width,
        label="No augmentation",
        color=colors,
        alpha=0.42,
        hatch="//",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean optimality gap (%)")
    axis.set_title("XML100: official EUC_2D gap")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, args.output_dir, "overall_gap", formats, args.dpi)
    plt.close(figure)

    # Empirical CDF of per-instance augmented gap.
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    for model, color in zip(models, colors):
        values = sorted(
            value
            for row in rows
            if row.get("model") == model and row.get("status") == "ok"
            if (value := finite_float(row, "gap_percent")) is not None
        )
        y = [(index + 1) / len(values) for index in range(len(values))]
        axis.plot(values, y, label=MODEL_LABELS.get(model, model), color=color, linewidth=1.8)
    axis.set_xlabel("Optimality gap (%)")
    axis.set_ylabel("Fraction of instances")
    axis.set_title("XML100 gap empirical CDF (8-fold)")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.005)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, args.output_dir, "gap_ecdf", formats, args.dpi)
    plt.close(figure)

    # Distribution view without drawing 40,000 outlier markers.
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    distributions = [
        [
            value
            for row in rows
            if row.get("model") == model and row.get("status") == "ok"
            if (value := finite_float(row, "gap_percent")) is not None
        ]
        for model in models
    ]
    try:
        boxes = axis.boxplot(
            distributions, tick_labels=labels, showfliers=False, patch_artist=True
        )
    except TypeError:  # matplotlib < 3.9
        boxes = axis.boxplot(distributions, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_ylabel("Optimality gap (%)")
    axis.set_title("XML100 gap distribution (outliers hidden)")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, args.output_dir, "gap_boxplot", formats, args.dpi)
    plt.close(figure)

    # Vehicle count is descriptive; it is not part of the XML100 objective.
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    vehicles = [mean_for(rows, model, "vehicles") for model in models]
    axis.bar(labels, vehicles, color=colors)
    axis.set_ylabel("Mean vehicles (reported only)")
    axis.set_title("XML100 route count")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, args.output_dir, "mean_vehicles", formats, args.dpi)
    plt.close(figure)

    # XML100 generator attributes: depot, customer, demand, and route-size type.
    attributes = [
        ("depot_type", "Depot"),
        ("customer_type", "Customers"),
        ("demand_type", "Demand"),
        ("route_size_type", "Average route size"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, attributes):
        categories = sorted({row[field] for row in rows if row.get(field)})
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            if row.get("status") != "ok":
                continue
            gap = finite_float(row, "gap_percent")
            if gap is not None:
                grouped[(row["model"], row[field])].append(gap)
        width = 0.8 / max(len(models), 1)
        base = list(range(len(categories)))
        for index, (model, color) in enumerate(zip(models, colors)):
            offset = (index - (len(models) - 1) / 2) * width
            values = [fmean(grouped[(model, category)]) for category in categories]
            axis.bar(
                [value + offset for value in base],
                values,
                width,
                label=MODEL_LABELS.get(model, model),
                color=color,
            )
        axis.set_xticks(base, categories, rotation=25, ha="right")
        axis.set_ylabel("Mean gap (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside upper center", ncol=len(models), frameon=False)
    save_figure(figure, args.output_dir, "attribute_gap", formats, args.dpi)
    plt.close(figure)

    print(f"wrote {5 * len(formats)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
