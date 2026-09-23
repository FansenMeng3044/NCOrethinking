"""Summarize matched Direct and Split MVMoE evaluation CSV files."""

import argparse
import csv
import math
import os
from collections import defaultdict


TRAINED = {"CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--trained_size", type=int, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    detail = []
    pools = defaultdict(lambda: {"direct": [], "split": []})
    for size in (50, 100, 200):
        direct_rows = read_rows(os.path.join(args.results_dir, f"direct_n{size}.csv"))
        split_rows = read_rows(os.path.join(args.results_dir, f"split_n{size}.csv"))
        direct = {(row["problem"], int(row["instance"])): row for row in direct_rows}
        split = {(row["problem"], int(row["instance"])): row for row in split_rows}
        if set(direct) != set(split):
            raise ValueError(f"Direct/Split instance keys differ for n={size}")

        for problem in sorted({key[0] for key in direct}):
            keys = sorted(key for key in direct if key[0] == problem)
            paired = [key for key in keys if split[key]["status"] == "ok"]
            direct_all = [float(direct[key]["cost"]) for key in keys]
            direct_paired = [float(direct[key]["cost"]) for key in paired]
            split_paired = [float(split[key]["cost"]) for key in paired]
            vehicles = [float(split[key]["vehicles"]) for key in paired]
            values = direct_all + direct_paired + split_paired + vehicles
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite value for n={size}, {problem}")
            direct_mean_all = sum(direct_all) / len(direct_all)
            if paired:
                direct_mean_paired = sum(direct_paired) / len(direct_paired)
                split_mean = sum(split_paired) / len(split_paired)
                delta = (split_mean / direct_mean_paired - 1.0) * 100.0
                vehicle_mean = sum(vehicles) / len(vehicles)
            else:
                direct_mean_paired = split_mean = delta = vehicle_mean = float("nan")
            regime = "trained_variant" if problem in TRAINED else "unseen_variant"
            detail.append(
                {
                    "model": args.model,
                    "trained_size": args.trained_size,
                    "evaluation_size": size,
                    "problem": problem,
                    "variant_regime": regime,
                    "instances": len(keys),
                    "direct_ok": len(keys),
                    "split_ok": len(paired),
                    "split_failed": len(keys) - len(paired),
                    "direct_mean_cost_all": f"{direct_mean_all:.10f}",
                    "direct_mean_cost_paired": f"{direct_mean_paired:.10f}",
                    "split_mean_cost_paired": f"{split_mean:.10f}",
                    "split_vs_direct_percent": f"{delta:.6f}",
                    "split_mean_vehicles": f"{vehicle_mean:.6f}",
                }
            )
            for key in paired:
                for group in ("all", regime):
                    pools[(size, group)]["direct"].append(float(direct[key]["cost"]))
                    pools[(size, group)]["split"].append(float(split[key]["cost"]))

    detail_path = os.path.join(args.results_dir, "comparison_by_environment.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)

    summary = []
    for (size, group), pool in sorted(pools.items()):
        direct_mean = sum(pool["direct"]) / len(pool["direct"])
        split_mean = sum(pool["split"]) / len(pool["split"])
        summary.append(
            {
                "model": args.model,
                "trained_size": args.trained_size,
                "evaluation_size": size,
                "variant_regime": group,
                "paired_instances": len(pool["direct"]),
                "direct_mean_cost": f"{direct_mean:.10f}",
                "split_mean_cost": f"{split_mean:.10f}",
                "split_vs_direct_percent": f"{(split_mean / direct_mean - 1.0) * 100.0:.6f}",
            }
        )
    summary_path = os.path.join(args.results_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
