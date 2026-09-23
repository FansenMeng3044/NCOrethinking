"""Validate completed Direct/Split evaluation artifacts."""

import argparse
import csv
import math
import os
from collections import Counter


PROBLEMS = {
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW",
}


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()

    for size in (50, 100, 200):
        for method in ("direct", "split"):
            path = os.path.join(args.results_dir, f"{method}_n{size}.csv")
            data = rows(path)
            assert len(data) == 16000, (path, len(data))
            keys = [(row["problem"], int(row["instance"])) for row in data]
            assert len(set(keys)) == 16000, f"duplicate keys in {path}"
            assert {key[0] for key in keys} == PROBLEMS
            counts = Counter(key[0] for key in keys)
            assert set(counts.values()) == {1000}, counts
            ok = 0
            for row in data:
                if row["status"] == "ok":
                    ok += 1
                    assert math.isfinite(float(row["cost"]))
                    if method == "split":
                        assert math.isfinite(float(row["vehicles"]))
                        assert float(row["vehicles"]) >= 1
                else:
                    assert method == "split"
                    assert row["status"] == "no_feasible_candidate"
                    assert row["cost"] == "" and row["vehicles"] == ""
            print(f"VALID {method} n={size}: rows=16000 ok={ok} failed={16000-ok}")

    detail = rows(os.path.join(args.results_dir, "comparison_by_environment.csv"))
    summary = rows(os.path.join(args.results_dir, "comparison_summary.csv"))
    assert len(detail) == 48, len(detail)
    assert len(summary) == 9, len(summary)
    print("VALID summaries: detail=48 summary=9")


if __name__ == "__main__":
    main()
