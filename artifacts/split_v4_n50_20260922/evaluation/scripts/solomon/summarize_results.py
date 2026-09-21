#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def number(value):
    return None if value in (None, "", "None", "nan") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["customers"], "ALL")].append(row)
        groups[(row["model"], row["customers"], row["family"])].append(row)

    output = []
    for (model, size, family), values in sorted(groups.items()):
        protocols = {row.get("protocol", "strict_fleet_lexicographic") for row in values}
        if len(protocols) != 1:
            raise ValueError(f"mixed protocols in {model}/{size}/{family}: {sorted(protocols)}")
        protocol = next(iter(protocols))
        status_counts = Counter(row["status"] for row in values)
        official_gaps = [number(row["sintef_official_distance_gap_pct"]) for row in values]
        official_gaps = [value for value in official_gaps if value is not None]
        distance_gaps = [number(row["sintef_nonofficial_distance_only_gap_pct"]) for row in values]
        distance_gaps = [value for value in distance_gaps if value is not None]
        legacy_gaps = [number(row["legacy_distance_only_gap_pct_noncomparable"]) for row in values]
        legacy_gaps = [value for value in legacy_gaps if value is not None]

        def mean(items):
            return sum(items) / len(items) if items else None

        def finite_column(name):
            items = [number(row[name]) for row in values]
            return [value for value in items if value is not None]

        successful = [row for row in values if row["status"] == "ok"]

        output.append(
            {
                "protocol": protocol,
                "model": model,
                "customers": size,
                "family": family,
                "instances": len(values),
                "successful_instances": len(successful),
                "no_feasible_instances": len(values) - len(successful),
                "status_counts": ";".join(
                    f"{status}:{count}" for status, count in sorted(status_counts.items())
                ),
                "mean_selected_vehicles": mean(finite_column("selected_vehicles")),
                "mean_selected_distance_raw": mean(finite_column("selected_distance_raw")),
                "selected_exceeds_reported_fleet_instances": sum(
                    row.get("selected_exceeds_reported_fleet", "").lower() == "true"
                    for row in values
                ),
                "mean_distance_only_vehicles": mean(finite_column("distance_only_vehicles")),
                "mean_distance_only_raw": mean(finite_column("distance_only_raw")),
                "mean_official_lex_vehicles": mean(finite_column("official_lex_vehicles")),
                "mean_official_lex_raw": mean(finite_column("official_lex_raw")),
                "same_candidate_count": sum(row["same_candidate"].lower() == "true" for row in values),
                "sintef_same_vehicle_instances": len(official_gaps),
                "sintef_mean_official_gap_pct_same_vehicle_only": mean(official_gaps),
                "sintef_mean_nonofficial_distance_only_gap_pct": mean(distance_gaps),
                "legacy_mean_gap_pct_noncomparable": mean(legacy_gaps),
                "all_candidates_feasible": all(
                    int(row["candidates_total"]) == int(row["candidates_externally_feasible"])
                    for row in values
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"saved {len(output)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
