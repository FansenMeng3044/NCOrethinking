#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from solomon_io import parse_instance, parse_solution, verify_routes


def main() -> None:
    parser = argparse.ArgumentParser(description="Double-precision Solomon VRPTW solution checker.")
    parser.add_argument("instance")
    parser.add_argument("solution")
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument(
        "--ignore-fleet-limit",
        action="store_true",
        help="match the MVMoE Table 7 distance protocol",
    )
    args = parser.parse_args()
    instance = parse_instance(args.instance)
    result = verify_routes(
        instance,
        parse_solution(args.solution),
        epsilon=args.epsilon,
        enforce_fleet_limit=not args.ignore_fleet_limit,
    )
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.feasible else 2)


if __name__ == "__main__":
    main()
