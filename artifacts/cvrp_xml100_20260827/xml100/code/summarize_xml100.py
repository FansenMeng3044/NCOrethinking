#!/usr/bin/env python3
"""Create rigorous XML100 aggregate, attribute, group, and paired summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty results file {path}")
    keys = [(row["model"], row["instance"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (model, instance) rows")
    return rows


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{row['model']}/{row['instance']}: non-finite {key}")
    return value


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def ci95_critical(count: int) -> tuple[float, str]:
    if count <= 1:
        return 0.0, "degenerate_single_observation"
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=count - 1)), "student_t"
    except ImportError:
        return NormalDist().inv_cdf(0.975), "normal_fallback_no_scipy"


def summarize_ok(rows: list[dict[str, str]]) -> dict[str, Any]:
    gaps = np.asarray([finite(row, "gap_percent") for row in rows], dtype=np.float64)
    costs = np.asarray([finite(row, "cost") for row in rows], dtype=np.float64)
    optima = np.asarray([finite(row, "optimum") for row in rows], dtype=np.float64)
    errors = costs - optima
    vehicles = np.asarray([finite(row, "vehicles") for row in rows], dtype=np.float64)
    inference = np.asarray(
        [finite(row, "inference_seconds_amortized") for row in rows], dtype=np.float64
    )
    postprocess = np.asarray(
        [finite(row, "postprocess_seconds") for row in rows], dtype=np.float64
    )
    count = len(rows)
    gap_std = float(gaps.std(ddof=1)) if count > 1 else 0.0
    gap_sem = gap_std / math.sqrt(count) if count else math.nan
    critical, ci_method = ci95_critical(count)
    return {
        "ok_count": count,
        "mean_cost": float(costs.mean()),
        "mean_optimum": float(optima.mean()),
        "mean_absolute_error": float(errors.mean()),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "gap_std_percent": gap_std,
        "gap_sem_percent": gap_sem,
        "gap_ci95_low_percent": float(gaps.mean() - critical * gap_sem),
        "gap_ci95_high_percent": float(gaps.mean() + critical * gap_sem),
        "gap_ci95_method": ci_method,
        "gap_p90_percent": percentile(gaps, 90),
        "gap_p95_percent": percentile(gaps, 95),
        "gap_p99_percent": percentile(gaps, 99),
        "gap_max_percent": float(gaps.max()),
        "optimal_hits": int(np.count_nonzero(errors == 0)),
        "mean_vehicles": float(vehicles.mean()),
        "mean_inference_seconds_amortized": float(inference.mean()),
        "mean_postprocess_seconds": float(postprocess.mean()),
        "throughput_instances_per_inference_second": (
            float(count / inference.sum()) if inference.sum() > 0 else math.nan
        ),
    }


def summarize_no_aug(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("no_aug_status") == "ok"]
    output: dict[str, Any] = {
        "no_aug_ok_count": len(ok),
        "no_aug_failure_count": len(rows) - len(ok),
        "no_aug_success_rate": len(ok) / len(rows),
    }
    if not ok:
        return output
    gaps = np.asarray([finite(row, "no_aug_gap_percent") for row in ok], dtype=np.float64)
    costs = np.asarray([finite(row, "no_aug_cost") for row in ok], dtype=np.float64)
    optima = np.asarray([finite(row, "optimum") for row in ok], dtype=np.float64)
    vehicles = np.asarray([finite(row, "no_aug_vehicles") for row in ok], dtype=np.float64)
    count = len(ok)
    standard_deviation = float(gaps.std(ddof=1)) if count > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(count)
    critical, ci_method = ci95_critical(count)
    output.update(
        {
            "no_aug_mean_cost": float(costs.mean()),
            "no_aug_mean_absolute_error": float((costs - optima).mean()),
            "no_aug_mean_gap_percent": float(gaps.mean()),
            "no_aug_median_gap_percent": float(np.median(gaps)),
            "no_aug_gap_std_percent": standard_deviation,
            "no_aug_gap_sem_percent": standard_error,
            "no_aug_gap_ci95_low_percent": float(gaps.mean() - critical * standard_error),
            "no_aug_gap_ci95_high_percent": float(gaps.mean() + critical * standard_error),
            "no_aug_gap_ci95_method": ci_method,
            "no_aug_gap_p95_percent": percentile(gaps, 95),
            "no_aug_gap_p99_percent": percentile(gaps, 99),
            "no_aug_gap_max_percent": float(gaps.max()),
            "no_aug_optimal_hits": int(np.count_nonzero(costs - optima == 0)),
            "no_aug_mean_vehicles": float(vehicles.mean()),
        }
    )
    return output


def grouped_summary(
    model_rows: list[dict[str, str]], dimension: str, value_key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in model_rows:
        grouped[row[value_key]].append(row)
    output: list[dict[str, Any]] = []
    for value, rows in sorted(grouped.items()):
        ok = [row for row in rows if row["status"] == "ok"]
        record: dict[str, Any] = {
            "model": rows[0]["model"],
            "architecture": rows[0]["architecture"],
            "training_size": int(rows[0]["training_size"]),
            "aggregation": dimension,
            "value": value,
            "expected_count": len(rows),
            "failure_count": len(rows) - len(ok),
            "success_rate": len(ok) / len(rows),
            "status_counts_json": json.dumps(
                {
                    status: sum(row["status"] == status for row in rows)
                    for status in sorted({row["status"] for row in rows})
                },
                sort_keys=True,
            ),
        }
        if ok:
            record.update(summarize_ok(ok))
        record.update(summarize_no_aug(rows))
        output.append(record)
    return output


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    adjusted = [math.nan] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def pairwise(
    rows: list[dict[str, str]],
    status_key: str = "status",
    gap_key: str = "gap_percent",
) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    all_models = sorted({row["model"] for row in rows})
    for row in rows:
        if row.get(status_key) == "ok":
            by_model[row["model"]][row["instance"]] = row
    models = all_models
    output: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            common = sorted(set(by_model[left]) & set(by_model[right]))
            left_gap = np.asarray(
                [finite(by_model[left][instance], gap_key) for instance in common]
            )
            right_gap = np.asarray(
                [finite(by_model[right][instance], gap_key) for instance in common]
            )
            differences = left_gap - right_gap
            if len(common) == 0:
                output.append(
                    {
                        "model_left": left,
                        "model_right": right,
                        "paired_instances": 0,
                        "left_wins": 0,
                        "ties": 0,
                        "right_wins": 0,
                        "mean_gap_difference_left_minus_right": "",
                        "median_gap_difference_left_minus_right": "",
                        "wilcoxon_two_sided_p": "",
                    }
                )
                raw_pvalues.append(math.nan)
                continue
            try:
                from scipy.stats import wilcoxon

                pvalue = (
                    1.0
                    if np.all(differences == 0)
                    else float(wilcoxon(differences, alternative="two-sided").pvalue)
                )
            except (ImportError, ValueError):
                pvalue = math.nan
            raw_pvalues.append(pvalue)
            output.append(
                {
                    "model_left": left,
                    "model_right": right,
                    "paired_instances": len(common),
                    "left_wins": int(np.count_nonzero(differences < 0)),
                    "ties": int(np.count_nonzero(differences == 0)),
                    "right_wins": int(np.count_nonzero(differences > 0)),
                    "mean_gap_difference_left_minus_right": float(differences.mean()),
                    "median_gap_difference_left_minus_right": float(np.median(differences)),
                    "wilcoxon_two_sided_p": pvalue if math.isfinite(pvalue) else "",
                }
            )
    finite_indices = [index for index, value in enumerate(raw_pvalues) if math.isfinite(value)]
    adjusted = holm_adjust([raw_pvalues[index] for index in finite_indices])
    for record in output:
        record["wilcoxon_holm_p"] = ""
    for index, value in zip(finite_indices, adjusted):
        output[index]["wilcoxon_holm_p"] = value
    return output


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty summary {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_overall_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# XML100 overall results",
        "",
        "Success is reported before conditional quality metrics.",
        "",
        "| Model | Architecture | Aug success | Aug mean Gap % | Aug P95 Gap % | "
        "No-aug success | No-aug mean Gap % | Mean vehicles | Inference s/instance | Postprocess s/instance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(records, key=lambda item: item["model"]):
        def number(key: str, digits: int = 4) -> str:
            value = record.get(key, "")
            return "" if value == "" else f"{float(value):.{digits}f}"

        lines.append(
            "| {model} | {architecture} | {ok}/{expected} | {gap} | {p95} | "
            "{no_ok}/{expected} | {no_gap} | {vehicles} | {inference} | {postprocess} |".format(
                model=record["model"],
                architecture=record["architecture"],
                ok=record.get("ok_count", 0),
                expected=record["expected_count"],
                gap=number("mean_gap_percent"),
                p95=number("gap_p95_percent"),
                no_ok=record.get("no_aug_ok_count", 0),
                no_gap=number("no_aug_mean_gap_percent"),
                vehicles=number("mean_vehicles", 3),
                inference=number("mean_inference_seconds_amortized", 6),
                postprocess=number("mean_postprocess_seconds", 6),
            )
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else results.parent
    )
    rows = read_rows(results)
    summary: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, model_rows in sorted(by_model.items()):
        overall = grouped_summary(model_rows, "overall", "dataset")
        if len(overall) != 1:
            raise AssertionError("one dataset expected")
        group_rows = grouped_summary(model_rows, "group", "group")
        ok_group_means = [
            float(record["mean_gap_percent"])
            for record in group_rows
            if record.get("ok_count") == record["expected_count"]
        ]
        overall[0]["macro_mean_of_complete_group_gaps"] = (
            float(np.mean(ok_group_means)) if ok_group_means else ""
        )
        overall[0]["complete_groups"] = len(ok_group_means)
        no_aug_group_means = [
            float(record["no_aug_mean_gap_percent"])
            for record in group_rows
            if record.get("no_aug_ok_count") == record["expected_count"]
        ]
        overall[0]["no_aug_macro_mean_of_complete_group_gaps"] = (
            float(np.mean(no_aug_group_means)) if no_aug_group_means else ""
        )
        overall[0]["no_aug_complete_groups"] = len(no_aug_group_means)
        summary.extend(overall)
        summary.extend(grouped_summary(model_rows, "depot", "depot_type"))
        summary.extend(grouped_summary(model_rows, "customer", "customer_type"))
        summary.extend(grouped_summary(model_rows, "demand", "demand_type"))
        summary.extend(grouped_summary(model_rows, "route_size", "route_size_type"))
        summary.extend(group_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "xml100_summary.csv", summary)
    write_overall_markdown(
        output_dir / "xml100_overall.md",
        [record for record in summary if record["aggregation"] == "overall"],
    )
    comparisons = pairwise(rows)
    if comparisons:
        atomic_csv(output_dir / "xml100_pairwise.csv", comparisons)
    no_aug_comparisons = pairwise(rows, "no_aug_status", "no_aug_gap_percent")
    if no_aug_comparisons:
        atomic_csv(output_dir / "xml100_pairwise_no_aug.csv", no_aug_comparisons)
    metadata = {
        "results": str(results),
        "rows": len(rows),
        "models": sorted(by_model),
        "summary_rows": len(summary),
        "pairwise_rows": len(comparisons),
        "no_aug_pairwise_rows": len(no_aug_comparisons),
        "primary_metric": (
            "report success rate first; when success rate is 100%, use the instance-weighted "
            "arithmetic mean optimality gap (%); otherwise the gap mean is conditional on success"
        ),
        "secondary_metrics": [
            "no-augmentation mean/median/tail optimality gaps",
            "median and tail gaps",
            "mean absolute cost error",
            "explicit success/failure counts",
            "macro mean across the 378 XML100 groups",
            "paired wins/ties/losses and two-sided Wilcoxon signed-rank with Holm correction",
        ],
        "warning": (
            "Mean raw cost is descriptive only and should not be used to rank methods across "
            "heterogeneous XML100 groups; use per-instance gap and paired comparisons."
        ),
    }
    temporary = output_dir / "xml100_summary_metadata.json.tmp"
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "xml100_summary_metadata.json")
    print(f"wrote {len(summary)} summary rows for {len(by_model)} models")


if __name__ == "__main__":
    main()
