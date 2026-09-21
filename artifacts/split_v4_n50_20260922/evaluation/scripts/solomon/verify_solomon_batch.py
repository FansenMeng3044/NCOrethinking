#!/usr/bin/env python
"""Independently replay every successful solution JSON in a Solomon result CSV."""

import argparse
import csv
import json
import math
from pathlib import Path

from solomon_io import parse_instance, verify_routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    results = Path(args.results).resolve()
    solutions = Path(args.solutions).resolve()
    data_root = Path(args.data_root).resolve()
    rows = list(csv.DictReader(results.open(encoding="utf-8", newline="")))
    if len(rows) != 224:
        raise ValueError(f"expected 224 rows (4 models x 56 instances), found {len(rows)}")
    keys = {(row["model"], row["instance"], int(row["customers"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate (model, instance, customers) result")

    verified = 0
    failures = []
    for row in rows:
        if row["status"] != "ok":
            failures.append({"key": [row["model"], row["instance"]], "status": row["status"]})
            continue
        size = int(row["customers"])
        if size != 50:
            raise ValueError(f"unexpected customer count {size}")
        instance = parse_instance(data_root / f"instances_{size}" / f"{row['instance']}.txt")
        solution_path = solutions / f"{row['model']}__{row['instance']}__n{size}.json"
        payload = json.loads(solution_path.read_text(encoding="utf-8"))
        selected = payload["selected_distance"]
        check = verify_routes(instance, selected["routes"], enforce_fleet_limit=False)
        expected_distance = float(row["selected_distance_raw"])
        if not check.feasible or not math.isclose(check.distance_raw, expected_distance, rel_tol=0, abs_tol=1e-8):
            failures.append({
                "key": [row["model"], row["instance"]],
                "feasible": check.feasible,
                "replayed_distance": check.distance_raw,
                "reported_distance": expected_distance,
            })
        else:
            verified += 1
    report = {
        "rows": len(rows),
        "unique_keys": len(keys),
        "verified_ok": verified,
        "failures": failures,
        "fleet_limit_enforced": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures or verified != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
