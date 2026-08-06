"""Independently audit POMO_SPLIT against exhaustive contiguous partitions.

The production SplitDecoder is the system under test.  This script deliberately
implements its oracle from scratch and does not import the repository's unit-test
helper.  It runs on CPU and is suitable for checking a saved run's src snapshot.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import sys
from pathlib import Path

import torch


CAPACITY_EPSILON = 1e-6


def load_split_module(source: Path):
    module_name = "pomo_split_decoder_under_audit"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SplitDecoder from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in ("split_giant_tours", "reconstruct_routes"):
        if not hasattr(module, name):
            raise AttributeError(f"{source} does not define {name}")
    return module


def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def exhaustive_split(depot, nodes, demands, tour, capacity):
    """Return the exact best cost over all contiguous cut patterns."""

    best_cost = math.inf
    best_routes = None
    n = len(tour)
    for cut_bits in itertools.product((False, True), repeat=max(0, n - 1)):
        boundaries = [0]
        boundaries.extend(index + 1 for index, cut in enumerate(cut_bits) if cut)
        boundaries.append(n)
        routes = [
            tour[boundaries[index] : boundaries[index + 1]]
            for index in range(len(boundaries) - 1)
        ]
        if any(
            sum(float(demands[customer - 1]) for customer in route)
            > capacity + CAPACITY_EPSILON
            for route in routes
        ):
            continue

        cost = 0.0
        for route in routes:
            previous = depot
            for customer in route:
                current = nodes[customer - 1]
                cost += distance(previous, current)
                previous = current
            cost += distance(previous, depot)

        if cost < best_cost:
            best_cost = cost
            best_routes = routes

    if best_routes is None:
        raise AssertionError("The exhaustive oracle found no feasible partition")
    return best_cost, best_routes


def check_reconstruction(tour, routes, demands, capacity, tolerance):
    flattened = [customer for route in routes for customer in route]
    if flattened != tour:
        raise AssertionError(
            f"Reconstructed routes changed the giant tour: {flattened} != {tour}"
        )
    for route in routes:
        load = sum(float(demands[customer - 1]) for customer in route)
        if load > capacity + tolerance:
            raise AssertionError(f"Reconstructed route load {load} exceeds {capacity}")


def compare_one(
    module,
    depot,
    nodes,
    demands,
    tour,
    capacity,
    tolerance,
):
    depot_tensor = torch.tensor([[depot]], dtype=torch.float32)
    node_tensor = torch.tensor([nodes], dtype=torch.float32)
    demand_tensor = torch.tensor([demands], dtype=torch.float32)
    tour_tensor = torch.tensor([[tour]], dtype=torch.long)

    result = module.split_giant_tours(
        depot_tensor,
        node_tensor,
        demand_tensor,
        tour_tensor,
        capacity=capacity,
        return_predecessors=True,
    )
    actual = float(result.costs[0, 0])
    expected, _ = exhaustive_split(depot, nodes, demands, tour, capacity)
    error = abs(actual - expected)
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=tolerance):
        raise AssertionError(
            f"Split mismatch: actual={actual:.12f}, expected={expected:.12f}, "
            f"abs_error={error:.3g}, tour={tour}, demands={demands}"
        )

    routes = module.reconstruct_routes(
        tour_tensor[0, 0], result.predecessors[0, 0]
    )
    check_reconstruction(tour, routes, demands, capacity, tolerance)
    return error, actual, routes


def run_edge_cases(module, tolerance):
    cases = [
        {
            "name": "single_customer_full_capacity",
            "depot": [0.0, 0.0],
            "nodes": [[3.0, 4.0]],
            "demands": [1.0],
            "tour": [1],
            "expected_cost": 10.0,
            "expected_routes": [[1]],
        },
        {
            "name": "exact_capacity_boundary",
            "depot": [0.0, 0.0],
            "nodes": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            "demands": [1.0, 0.4, 0.6],
            "tour": [1, 2, 3],
            "expected_cost": 8.0,
            "expected_routes": [[1], [2, 3]],
        },
        {
            "name": "all_customers_one_route",
            "depot": [0.0, 0.0],
            "nodes": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            "demands": [0.1, 0.1, 0.1],
            "tour": [1, 2, 3],
            "expected_cost": 6.0,
            "expected_routes": [[1, 2, 3]],
        },
        {
            "name": "every_customer_separate",
            "depot": [0.0, 0.0],
            "nodes": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            "demands": [0.6, 0.6, 0.6],
            "tour": [1, 2, 3],
            "expected_cost": 12.0,
            "expected_routes": [[1], [2], [3]],
        },
    ]

    results = []
    for case in cases:
        error, actual, routes = compare_one(
            module,
            case["depot"],
            case["nodes"],
            case["demands"],
            case["tour"],
            1.0,
            tolerance,
        )
        if not math.isclose(
            actual, case["expected_cost"], rel_tol=1e-6, abs_tol=tolerance
        ):
            raise AssertionError(
                f"{case['name']} cost {actual} != {case['expected_cost']}"
            )
        if routes != case["expected_routes"]:
            raise AssertionError(
                f"{case['name']} routes {routes} != {case['expected_routes']}"
            )
        results.append(
            {
                "name": case["name"],
                "cost": actual,
                "routes": routes,
                "abs_error_vs_oracle": error,
            }
        )
    return results


def run_random_cases(module, seed, max_n, cases_per_n, batch, pomo, tolerance):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    comparisons = 0
    max_error = 0.0
    worst_case = None

    for n in range(1, max_n + 1):
        for case_index in range(cases_per_n):
            depots = torch.rand(batch, 1, 2, generator=generator)
            nodes = torch.rand(batch, n, 2, generator=generator)
            demands = torch.randint(1, 10, (batch, n), generator=generator).float() / 10
            tours = torch.stack(
                [
                    torch.stack(
                        [torch.randperm(n, generator=generator) + 1 for _ in range(pomo)]
                    )
                    for _ in range(batch)
                ]
            )
            result = module.split_giant_tours(
                depots,
                nodes,
                demands,
                tours,
                capacity=1.0,
                return_predecessors=True,
            )

            for batch_index in range(batch):
                depot = depots[batch_index, 0].tolist()
                node_list = nodes[batch_index].tolist()
                demand_list = demands[batch_index].tolist()
                for pomo_index in range(pomo):
                    tour = tours[batch_index, pomo_index].tolist()
                    expected, _ = exhaustive_split(
                        depot, node_list, demand_list, tour, 1.0
                    )
                    actual = float(result.costs[batch_index, pomo_index])
                    error = abs(actual - expected)
                    if error > max_error:
                        max_error = error
                        worst_case = {
                            "n": n,
                            "case_index": case_index,
                            "batch_index": batch_index,
                            "pomo_index": pomo_index,
                            "actual": actual,
                            "expected": expected,
                        }
                    if not math.isclose(
                        actual, expected, rel_tol=1e-6, abs_tol=tolerance
                    ):
                        raise AssertionError(
                            f"Random mismatch at n={n}, case={case_index}, "
                            f"batch={batch_index}, pomo={pomo_index}: "
                            f"actual={actual:.12f}, expected={expected:.12f}, "
                            f"abs_error={error:.3g}"
                        )
                    routes = module.reconstruct_routes(
                        tours[batch_index, pomo_index],
                        result.predecessors[batch_index, pomo_index],
                    )
                    check_reconstruction(
                        tour, routes, demand_list, 1.0, tolerance
                    )
                    comparisons += 1

    return {
        "comparisons": comparisons,
        "max_abs_error": max_error,
        "worst_case": worst_case,
    }


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit POMO_SPLIT against an independent exhaustive oracle."
    )
    parser.add_argument("--pomo-root", type=Path, required=True)
    parser.add_argument(
        "--split-source",
        type=Path,
        help="SplitDecoder.py to audit; defaults to the current POMO_SPLIT source.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--cases-per-n", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--pomo", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.split_source
    if source is None:
        source = (
            args.pomo_root
            / "NEW_py_ver"
            / "CVRP"
            / "POMO_SPLIT"
            / "SplitDecoder.py"
        )
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.max_n < 1 or args.max_n > 15:
        raise ValueError("--max-n must be in 1..15; exhaustive work is exponential")
    if min(args.cases_per_n, args.batch, args.pomo) < 1:
        raise ValueError("case, batch, and POMO counts must be positive")

    module = load_split_module(source)
    edge_results = run_edge_cases(module, args.tolerance)
    random_results = run_random_cases(
        module,
        args.seed,
        args.max_n,
        args.cases_per_n,
        args.batch,
        args.pomo,
        args.tolerance,
    )
    report = {
        "status": "passed",
        "split_source": str(source),
        "split_source_sha256": sha256_file(source),
        "seed": args.seed,
        "max_n": args.max_n,
        "cases_per_n": args.cases_per_n,
        "batch": args.batch,
        "pomo": args.pomo,
        "tolerance": args.tolerance,
        "edge_cases": edge_results,
        "random": random_results,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
