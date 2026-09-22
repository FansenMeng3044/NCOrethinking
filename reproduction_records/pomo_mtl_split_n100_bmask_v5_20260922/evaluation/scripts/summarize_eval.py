"""Summarize matched Direct and Split evaluation CSV files."""

import argparse
import csv
import math
import os
from collections import defaultdict


TRAINED = {"CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(values):
    return all(math.isfinite(value) for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    detail = []
    paired_pool = defaultdict(lambda: {"direct": [], "split": []})
    for size in (50, 100, 200):
        direct_rows = read_rows(os.path.join(args.results_dir, f"direct_n{size}.csv"))
        split_rows = read_rows(os.path.join(args.results_dir, f"split_n{size}.csv"))
        direct = {(r["problem"], int(r["instance"])): r for r in direct_rows}
        split = {(r["problem"], int(r["instance"])): r for r in split_rows}
        if set(direct) != set(split):
            raise ValueError(f"Direct/Split keys differ for n={size}")

        for problem in sorted({key[0] for key in direct}):
            keys = sorted(key for key in direct if key[0] == problem)
            direct_costs = [float(direct[key]["cost"]) for key in keys]
            paired_keys = [key for key in keys if split[key]["status"] == "ok"]
            split_costs = [float(split[key]["cost"]) for key in paired_keys]
            direct_paired = [float(direct[key]["cost"]) for key in paired_keys]
            vehicles = [float(split[key]["vehicles"]) for key in paired_keys]
            if not finite(direct_costs + split_costs + direct_paired + vehicles):
                raise ValueError(f"non-finite result for n={size}, {problem}")
            direct_mean = sum(direct_costs) / len(direct_costs)
            if paired_keys:
                direct_pair_mean = sum(direct_paired) / len(direct_paired)
                split_mean = sum(split_costs) / len(split_costs)
                delta = (split_mean / direct_pair_mean - 1.0) * 100.0
                vehicle_mean = sum(vehicles) / len(vehicles)
            else:
                direct_pair_mean = split_mean = delta = vehicle_mean = float("nan")
            regime = "trained_variant" if problem in TRAINED else "unseen_variant"
            detail.append({
                "problem_size": size,
                "problem": problem,
                "variant_regime": regime,
                "instances": len(keys),
                "direct_ok": len(keys),
                "split_ok": len(paired_keys),
                "split_failed": len(keys) - len(paired_keys),
                "direct_mean_cost_all": f"{direct_mean:.10f}",
                "direct_mean_cost_paired": f"{direct_pair_mean:.10f}",
                "split_mean_cost_paired": f"{split_mean:.10f}",
                "split_vs_direct_percent": f"{delta:.6f}",
                "split_mean_vehicles": f"{vehicle_mean:.6f}",
            })
            for key in paired_keys:
                for group in ("all", regime):
                    pool = paired_pool[(size, group)]
                    pool["direct"].append(float(direct[key]["cost"]))
                    pool["split"].append(float(split[key]["cost"]))

    detail_path = os.path.join(args.results_dir, "comparison_by_environment.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)

    groups = []
    for (size, group), pool in sorted(paired_pool.items()):
        direct_mean = sum(pool["direct"]) / len(pool["direct"])
        split_mean = sum(pool["split"]) / len(pool["split"])
        groups.append({
            "problem_size": size,
            "variant_regime": group,
            "paired_instances": len(pool["direct"]),
            "direct_mean_cost": f"{direct_mean:.10f}",
            "split_mean_cost": f"{split_mean:.10f}",
            "split_vs_direct_percent": f"{(split_mean / direct_mean - 1.0) * 100.0:.6f}",
        })
    group_path = os.path.join(args.results_dir, "comparison_summary.csv")
    with open(group_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=groups[0].keys())
        writer.writeheader()
        writer.writerows(groups)
    print(detail_path)
    print(group_path)


if __name__ == "__main__":
    main()
