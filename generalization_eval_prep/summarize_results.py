#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "architecture": summary["architecture"],
                "training_size": summary["training_size"],
                "test_size": summary["problem_size"],
                "instances": summary["instances"],
                "success": summary["success"],
                "mean_cost": summary["mean_cost"],
                "mean_vehicles": summary["mean_vehicles"],
                "mean_inference_seconds": summary["mean_inference_seconds"],
                "pomo_starts": summary["pomo_starts"],
                "augmentation": summary["augmentation"],
                "checkpoint_sha256": summary["checkpoint"]["sha256"],
                "dataset_sha256": summary["dataset_sha256"],
            }
        )
    path = args.root / "aggregate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
