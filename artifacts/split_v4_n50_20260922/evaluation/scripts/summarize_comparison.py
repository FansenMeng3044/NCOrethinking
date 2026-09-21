#!/usr/bin/env python
"""Create one Direct-versus-Split report across all completed n=50 evaluations."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/root/autodl-tmp/split_v4_n50_eval_20260922")
RESULTS = ROOT / "results"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values):
    return all(math.isfinite(float(value)) for value in values)


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values)


def pair(dataset, backbone, direct_rows, split_rows, cost_key, vehicle_key=None):
    direct = {row["instance"]: row for row in direct_rows}
    split = {row["instance"]: row for row in split_rows}
    if set(direct) != set(split):
        raise ValueError(f"{dataset}/{backbone}: paired instance keys differ")
    keys = sorted(direct)
    direct_cost = [float(direct[key][cost_key]) for key in keys]
    split_cost = [float(split[key][cost_key]) for key in keys]
    if not finite(direct_cost + split_cost):
        raise ValueError(f"{dataset}/{backbone}: non-finite paired cost")
    direct_better = sum(a < b - 1e-9 for a, b in zip(direct_cost, split_cost))
    split_better = sum(b < a - 1e-9 for a, b in zip(direct_cost, split_cost))
    row = {
        "dataset": dataset,
        "backbone": backbone,
        "instances": len(keys),
        "direct_mean_cost": mean(direct_cost),
        "split_mean_cost": mean(split_cost),
        "split_relative_change_percent": 100 * (mean(split_cost) / mean(direct_cost) - 1),
        "direct_better_instances": direct_better,
        "split_better_instances": split_better,
        "ties": len(keys) - direct_better - split_better,
        "direct_mean_vehicles": "",
        "split_mean_vehicles": "",
    }
    if vehicle_key:
        row["direct_mean_vehicles"] = mean(direct[key][vehicle_key] for key in keys)
        row["split_mean_vehicles"] = mean(split[key][vehicle_key] for key in keys)
    return row


def add_overall(rows, dataset, backbone, variant, model, records, cost_key, vehicle_key=None, gap_key=None):
    if not records:
        raise ValueError(f"{dataset}/{model}: empty result")
    costs = [float(row[cost_key]) for row in records]
    if not finite(costs):
        raise ValueError(f"{dataset}/{model}: non-finite costs")
    rows.append({
        "dataset": dataset,
        "backbone": backbone,
        "variant": variant,
        "model": model,
        "instances": len(records),
        "success": sum(row.get("status", "ok") == "ok" for row in records),
        "mean_cost": mean(costs),
        "mean_gap_percent": mean(row[gap_key] for row in records) if gap_key else "",
        "mean_vehicles": mean(row[vehicle_key] for row in records) if vehicle_key else "",
    })


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    overall, paired = [], []

    fixed_cvrp = {}
    for model in ("am_direct_n50", "am_split_v4_n50", "pomo_direct_n50", "pomo_split_v4_n50"):
        rows = read_csv(RESULTS / "fixed_cvrp" / f"{model}.csv")
        if len(rows) != 10000:
            raise ValueError(f"{model}: expected 10000 fixed CVRP rows")
        fixed_cvrp[model] = rows
        backbone = "AM" if model.startswith("am_") else "POMO"
        variant = "Split" if "split" in model else "Direct"
        add_overall(overall, "Frozen CVRP50", backbone, variant, model, rows, "distance_aug")
    paired.append(pair("Frozen CVRP50", "AM", fixed_cvrp["am_direct_n50"], fixed_cvrp["am_split_v4_n50"], "distance_aug"))
    paired.append(pair("Frozen CVRP50", "POMO", fixed_cvrp["pomo_direct_n50"], fixed_cvrp["pomo_split_v4_n50"], "distance_aug"))

    fixed_tw = {}
    for model in ("am_tw_direct_n50", "am_tw_split_v4_n50", "pomo_tw_direct_n50", "pomo_tw_split_v4_n50"):
        rows = read_csv(RESULTS / "fixed_tw" / f"{model}.csv")
        if len(rows) != 10000 or any(row["feasible"] != "1" for row in rows):
            raise ValueError(f"{model}: fixed TW completeness/feasibility failure")
        fixed_tw[model] = rows
        backbone = "AM" if model.startswith("am_") else "POMO"
        variant = "Split" if "split" in model else "Direct"
        add_overall(overall, "Frozen CVRPTW50", backbone, variant, model, rows, "distance_aug", "vehicle_count")
    paired.append(pair("Frozen CVRPTW50", "AM", fixed_tw["am_tw_direct_n50"], fixed_tw["am_tw_split_v4_n50"], "distance_aug", "vehicle_count"))
    paired.append(pair("Frozen CVRPTW50", "POMO", fixed_tw["pomo_tw_direct_n50"], fixed_tw["pomo_tw_split_v4_n50"], "distance_aug", "vehicle_count"))

    xml_rows = read_csv(RESULTS / "xml100" / "xml100_results.csv")
    if len(xml_rows) != 40000 or any(row["status"] != "ok" for row in xml_rows):
        raise ValueError("XML100 completeness/status failure")
    xml_by_model = defaultdict(list)
    for row in xml_rows:
        xml_by_model[row["model"]].append(row)
    for model, backbone, variant in (
        ("am_direct_n50", "AM", "Direct"),
        ("am_split_v4_n50", "AM", "Split"),
        ("pomo_direct_n50", "POMO", "Direct"),
        ("pomo_split_v4_n50", "POMO", "Split"),
    ):
        add_overall(overall, "CVRPLIB XML100", backbone, variant, model, xml_by_model[model], "cost", "vehicles", "gap_percent")
    paired.append(pair("CVRPLIB XML100", "AM", xml_by_model["am_direct_n50"], xml_by_model["am_split_v4_n50"], "cost", "vehicles"))
    paired.append(pair("CVRPLIB XML100", "POMO", xml_by_model["pomo_direct_n50"], xml_by_model["pomo_split_v4_n50"], "cost", "vehicles"))

    solomon_rows = read_csv(RESULTS / "solomon50" / "solomon_results.csv")
    if len(solomon_rows) != 224 or any(row["status"] != "ok" for row in solomon_rows):
        raise ValueError("Solomon50 completeness/status failure")
    solomon_by_model = defaultdict(list)
    for row in solomon_rows:
        solomon_by_model[row["model"]].append(row)
    for model, backbone, variant in (
        ("am_tw_direct_n50", "AM", "Direct"),
        ("am_tw_split_v4_n50", "AM", "Split"),
        ("pomo_tw_direct_n50", "POMO", "Direct"),
        ("pomo_tw_split_v4_n50", "POMO", "Split"),
    ):
        add_overall(overall, "Solomon50", backbone, variant, model, solomon_by_model[model], "selected_distance_raw", "selected_vehicles")
    paired.append(pair("Solomon50", "AM", solomon_by_model["am_tw_direct_n50"], solomon_by_model["am_tw_split_v4_n50"], "selected_distance_raw", "selected_vehicles"))
    paired.append(pair("Solomon50", "POMO", solomon_by_model["pomo_tw_direct_n50"], solomon_by_model["pomo_tw_split_v4_n50"], "selected_distance_raw", "selected_vehicles"))

    family_rows = []
    for model, records in solomon_by_model.items():
        backbone = "AM" if model.startswith("am_") else "POMO"
        variant = "Split" if "split" in model else "Direct"
        by_family = defaultdict(list)
        for row in records:
            by_family[row["family"]].append(row)
        for family in ("C1", "C2", "R1", "R2", "RC1", "RC2"):
            group = by_family[family]
            family_rows.append({
                "family": family,
                "backbone": backbone,
                "variant": variant,
                "model": model,
                "instances": len(group),
                "success": sum(row["status"] == "ok" for row in group),
                "mean_distance": mean(row["selected_distance_raw"] for row in group),
                "mean_vehicles": mean(row["selected_vehicles"] for row in group),
                "mean_elapsed_seconds": mean(row["elapsed_seconds"] for row in group),
            })

    write_csv(RESULTS / "comparison_overall.csv", overall)
    write_csv(RESULTS / "comparison_paired.csv", paired)
    write_csv(RESULTS / "solomon_family_comparison.csv", family_rows)
    report = {
        "overall": overall,
        "paired": paired,
        "validation": {
            "fixed_cvrp_rows": 40000,
            "fixed_tw_rows": 40000,
            "fixed_tw_feasible": 40000,
            "xml100_rows": 40000,
            "xml100_verified": 40000,
            "solomon_rows": 224,
            "solomon_verified": 224,
        },
    }
    (RESULTS / "comparison_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
