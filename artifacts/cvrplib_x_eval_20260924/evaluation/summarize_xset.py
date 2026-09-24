#!/usr/bin/env python3
"""Summarize and independently recheck a completed X-set evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from xset_io import parse_instance, parse_solution, verify_routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()

    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted((output / "solutions").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = parse_instance(data / f"{payload['instance']}.vrp")
        reference = parse_solution(data / f"{payload['instance']}.sol")
        reference_check = verify_routes(instance, reference.routes)
        if not reference_check.feasible or reference_check.cost != reference.declared_cost:
            raise ValueError(f"reference solution failed verification: {instance.name}")
        if payload["status"] != "ok":
            raise ValueError(f"unsuccessful model result: {path}")
        check = verify_routes(instance, payload["routes"])
        if not check.feasible or check.cost != payload["cost"]:
            raise ValueError(f"model result failed verification: {path}")
        expected_gap = 100.0 * (payload["cost"] - reference.declared_cost) / reference.declared_cost
        if abs(expected_gap - payload["gap_percent"]) > 1e-9:
            raise ValueError(f"gap mismatch: {path}")
        row = {
            "model": payload["model"],
            "instance": payload["instance"],
            "customers": instance.customers,
            "cost": payload["cost"],
            "reference_cost": reference.declared_cost,
            "gap_percent": payload["gap_percent"],
            "vehicles": payload["vehicles"],
            "inference_seconds": payload["inference_seconds_amortized"],
            "postprocess_seconds": payload["postprocess_seconds"],
            "checkpoint_sha256": payload["checkpoint"]["sha256"],
        }
        rows.append(row)
        grouped[str(payload["model"])].append(row)

    if not rows:
        raise ValueError("no result JSON files found")
    csv_path = output / "xset_verified_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model, items in sorted(grouped.items()):
        if len(items) != 100:
            raise ValueError(f"{model}: expected 100 instances, found {len(items)}")
        summary.append(
            {
                "model": model,
                "instances": len(items),
                "mean_gap_percent": statistics.fmean(float(x["gap_percent"]) for x in items),
                "median_gap_percent": statistics.median(float(x["gap_percent"]) for x in items),
                "mean_vehicles": statistics.fmean(float(x["vehicles"]) for x in items),
                "total_inference_seconds": sum(float(x["inference_seconds"]) for x in items),
                "total_postprocess_seconds": sum(float(x["postprocess_seconds"]) for x in items),
            }
        )
    (output / "xset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
