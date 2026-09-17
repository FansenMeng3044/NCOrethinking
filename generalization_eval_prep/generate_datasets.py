#!/usr/bin/env python3
"""Generate deterministic out-of-size CVRP test sets for n=200/500/1000.

The training-size-100 distribution is held fixed: coordinates are uniform in
[0, 1]^2 and integer demands are uniform in {1, ..., 9} with vehicle capacity
50.  Keeping capacity 50 isolates size generalization instead of changing both
problem size and the demand/capacity distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(200, 500, 1000))
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--capacity", type=int, default=50)
    args = parser.parse_args()
    if args.instances <= 0 or args.capacity <= 0:
        raise ValueError("instances and capacity must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for size in args.sizes:
        if size <= 0:
            raise ValueError("all sizes must be positive")
        # Use an independent generator with the same documented seed for each
        # dataset, matching the conventional *_test_seed1234 naming scheme.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        depot_xy = torch.rand(args.instances, 1, 2, generator=generator)
        node_xy = torch.rand(args.instances, size, 2, generator=generator)
        demand_int = torch.randint(
            1, 10, (args.instances, size), generator=generator, dtype=torch.int16
        )
        node_demand = demand_int.float() / float(args.capacity)
        payload = {
            "depot_xy": depot_xy,
            "node_xy": node_xy,
            "node_demand": node_demand,
            "node_demand_integer": demand_int,
            "vehicle_capacity_integer": args.capacity,
            "seed": args.seed,
            "generator": "torch_cpu_uniform_xy_randint_1_9_capacity50_v1",
        }
        path = args.output / f"cvrp{size}_n{args.instances}_seed{args.seed}_cap{args.capacity}.pt"
        torch.save(payload, path)
        manifest.append(
            {
                "path": path.name,
                "problem": "CVRP",
                "customers": size,
                "instances": args.instances,
                "seed": args.seed,
                "coordinate_distribution": "iid Uniform[0,1)^2",
                "demand_distribution": "iid discrete Uniform{1,...,9}",
                "vehicle_capacity_integer": args.capacity,
                "normalized_vehicle_capacity": 1.0,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
