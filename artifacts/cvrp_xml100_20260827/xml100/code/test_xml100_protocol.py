#!/usr/bin/env python3
"""Protocol-level regression tests; no neural checkpoint is required."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from evaluate_xml100 import select_depot_actions
from verify_xml100_results import independently_verify_routes
from xml100_io import (
    CVRPInstance,
    parse_instance,
    parse_solution,
    parse_xml100_name,
    route_cost,
    split_giant_tours_euc2d,
    verify_routes,
)


HERE = Path(__file__).resolve().parent


def synthetic(coords: list[tuple[int, int]], demands: list[int], capacity: int) -> CVRPInstance:
    coordinate_array = np.asarray(coords, dtype=np.int64)
    demand_array = np.asarray(demands, dtype=np.int64)
    coordinate_array.setflags(write=False)
    demand_array.setflags(write=False)
    return CVRPInstance(
        "XML100_1111_01",
        capacity,
        coordinate_array,
        demand_array,
        Path("synthetic.vrp"),
        parse_xml100_name("XML100_1111_01"),
    )


class XML100ProtocolTests(unittest.TestCase):
    def test_per_edge_rounding_is_not_total_rounding(self) -> None:
        instance = synthetic([(0, 0), (1, 1), (2, 0)], [0, 1, 1], 2)
        official = route_cost(instance, [[1, 2]])
        continuous_total_rounded = round(math.sqrt(2) + math.sqrt(2) + 2)
        self.assertEqual(official, 4)
        self.assertEqual(continuous_total_rounded, 5)

    def test_direct_actions_have_implicit_depot_at_both_ends(self) -> None:
        instance = synthetic([(0, 0), (10, 0), (0, 10)], [0, 1, 1], 1)
        selection = select_depot_actions(
            instance,
            np.asarray([[1, 0, 2]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        )
        self.assertEqual(selection.routes, [[1], [2]])
        self.assertEqual(selection.cost, 40)

    def test_exact_split_matches_exhaustive_partitioning(self) -> None:
        instance = synthetic(
            [(0, 0), (10, 0), (11, 0), (0, 10), (0, 11)],
            [0, 2, 2, 2, 2],
            4,
        )
        tour = [1, 2, 3, 4]
        routes, costs = split_giant_tours_euc2d(instance, np.asarray([tour]))
        best = None
        for mask in range(1 << (len(tour) - 1)):
            candidate: list[list[int]] = []
            current = [tour[0]]
            for edge in range(len(tour) - 1):
                if mask & (1 << edge):
                    candidate.append(current)
                    current = []
                current.append(tour[edge + 1])
            candidate.append(current)
            verification = verify_routes(instance, candidate)
            if verification.feasible:
                best = verification.cost if best is None else min(best, verification.cost)
        self.assertEqual(int(costs[0]), best)
        self.assertEqual(route_cost(instance, routes[0]), best)

    def test_independent_integer_verifier_agrees_on_valid_route(self) -> None:
        instance = synthetic([(0, 0), (3, 4), (6, 8)], [0, 1, 1], 2)
        feasible, cost, vehicles, errors = independently_verify_routes(instance, [[1, 2]])
        self.assertTrue(feasible, errors)
        self.assertEqual(cost, route_cost(instance, [[1, 2]]))
        self.assertEqual(vehicles, 1)

    def test_release_reference_anomaly_is_detected_not_repaired(self) -> None:
        instances = HERE / "data" / "XML" / "instances"
        solutions = HERE / "data" / "XML" / "solutions"
        if not instances.is_dir() or not solutions.is_dir():
            self.skipTest("official XML100 release is not extracted")
        instance = parse_instance(instances / "XML100_1121_04.vrp")
        solution = parse_solution(solutions / "XML100_1121_04.sol")
        verification = verify_routes(instance, solution.routes)
        self.assertFalse(verification.feasible)
        self.assertIn("duplicate customers: [33]", verification.errors)
        self.assertEqual(solution.declared_cost, 24425)

    def test_normal_release_route_matches_integer_objective(self) -> None:
        instances = HERE / "data" / "XML" / "instances"
        solutions = HERE / "data" / "XML" / "solutions"
        if not instances.is_dir() or not solutions.is_dir():
            self.skipTest("official XML100 release is not extracted")
        instance = parse_instance(instances / "XML100_1111_01.vrp")
        solution = parse_solution(solutions / "XML100_1111_01.sol")
        verification = verify_routes(instance, solution.routes)
        self.assertTrue(verification.feasible, verification.errors)
        self.assertEqual(verification.cost, solution.declared_cost)


if __name__ == "__main__":
    unittest.main(verbosity=2)
