#!/usr/bin/env python
"""Generate a canonical fixed CVRPTW tensor dataset for all four models."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from CVRPTWCore import get_random_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--depot-start", type=float, default=0.0)
    parser.add_argument("--depot-end", type=float, default=3.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--service-duration", type=float, default=0.2)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    problems = get_random_problems(
        args.samples,
        args.problem_size,
        device="cpu",
        depot_start=args.depot_start,
        depot_end=args.depot_end,
        speed=args.speed,
        service_duration=args.service_duration,
    )
    names = (
        "depot_xy", "node_xy", "node_demand", "node_service_time",
        "node_tw_start", "node_tw_end",
    )
    payload = {name: value.cpu() for name, value in zip(names, problems)}
    payload.update(
        {
            "problem": "cvrptw",
            "problem_size": args.problem_size,
            "seed": args.seed,
            "capacity": 1.0,
            "depot_start": args.depot_start,
            "depot_end": args.depot_end,
            "speed": args.speed,
            "service_duration": args.service_duration,
            "semantics": "multi_vehicle_clock_reset_at_depot",
            "objective": "total_euclidean_distance",
        }
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print("saved", output)
    print("sha256", digest)


if __name__ == "__main__":
    main()
