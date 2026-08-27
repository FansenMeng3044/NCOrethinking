#!/usr/bin/env python3
"""Strict CVRPLIB XML100 parsing, scoring, Split, and verification.

Customer IDs exposed by this module are 1..n, matching CVRPLIB solution
files.  Internally index 0 is the depot and indices 1..n are customers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


XML100_RE = re.compile(r"^XML100_([123])([123])([1-7])([1-6])_(\d{2})$")

DEPOT_LABELS = {"1": "random", "2": "centered", "3": "cornered"}
CUSTOMER_LABELS = {"1": "random", "2": "clustered", "3": "random_clustered"}
DEMAND_LABELS = {
    "1": "unitary",
    "2": "small_large_cv",
    "3": "small_small_cv",
    "4": "large_large_cv",
    "5": "large_small_cv",
    "6": "quadrant_dependent",
    "7": "many_small_few_large",
}
ROUTE_SIZE_LABELS = {
    "1": "very_short",
    "2": "short",
    "3": "medium",
    "4": "long",
    "5": "very_long",
    "6": "ultra_long",
}


@dataclass(frozen=True)
class XML100Features:
    depot_code: str
    customer_code: str
    demand_code: str
    route_size_code: str
    instance_id: str

    @property
    def group(self) -> str:
        return (
            self.depot_code
            + self.customer_code
            + self.demand_code
            + self.route_size_code
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "group": self.group,
            "depot_code": self.depot_code,
            "depot_type": DEPOT_LABELS[self.depot_code],
            "customer_code": self.customer_code,
            "customer_type": CUSTOMER_LABELS[self.customer_code],
            "demand_code": self.demand_code,
            "demand_type": DEMAND_LABELS[self.demand_code],
            "route_size_code": self.route_size_code,
            "route_size_type": ROUTE_SIZE_LABELS[self.route_size_code],
            "instance_id": self.instance_id,
        }


@dataclass(frozen=True)
class CVRPInstance:
    name: str
    capacity: int
    coords: np.ndarray  # (n + 1, 2), depot first, int64
    demands: np.ndarray  # (n + 1,), depot first, int64
    source_path: Path
    features: XML100Features

    @property
    def customers(self) -> int:
        return int(self.coords.shape[0] - 1)

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        # TSPLIB/CVRPLIB EUC_2D: nint(sqrt(dx^2 + dy^2)) for every edge.
        delta = self.coords[:, None, :] - self.coords[None, :, :]
        euclidean = np.sqrt(np.square(delta, dtype=np.int64).sum(axis=2))
        return np.floor(euclidean + 0.5).astype(np.int64)


@dataclass(frozen=True)
class ParsedSolution:
    routes: list[list[int]]
    declared_cost: int
    source_path: Path


@dataclass(frozen=True)
class Verification:
    feasible: bool
    cost: int | None
    vehicles: int
    errors: tuple[str, ...]


def parse_xml100_name(name: str) -> XML100Features:
    match = XML100_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not an official XML100 name: {name}")
    return XML100Features(*match.groups())


def _header_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip().upper(), value.strip()


def parse_instance(path: str | Path) -> CVRPInstance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    headers: dict[str, str] = {}
    coords_by_id: dict[int, tuple[int, int]] = {}
    demands_by_id: dict[int, int] = {}
    depot_ids: list[int] = []
    section: str | None = None

    for line in lines:
        if not line:
            continue
        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            section = "coords"
            continue
        if upper == "DEMAND_SECTION":
            section = "demands"
            continue
        if upper == "DEPOT_SECTION":
            section = "depot"
            continue
        if upper == "EOF":
            break

        if section is None:
            item = _header_value(line)
            if item is not None:
                headers[item[0]] = item[1]
            continue
        parts = line.split()
        if section == "coords":
            if len(parts) != 3:
                raise ValueError(f"{path}: malformed NODE_COORD_SECTION line: {line}")
            node_id, x, y = map(int, parts)
            if node_id in coords_by_id:
                raise ValueError(f"{path}: duplicate coordinate node ID {node_id}")
            coords_by_id[node_id] = (x, y)
        elif section == "demands":
            if len(parts) != 2:
                raise ValueError(f"{path}: malformed DEMAND_SECTION line: {line}")
            node_id, demand = map(int, parts)
            if node_id in demands_by_id:
                raise ValueError(f"{path}: duplicate demand node ID {node_id}")
            demands_by_id[node_id] = demand
        elif section == "depot":
            depot_id = int(parts[0])
            if depot_id == -1:
                section = "done"
            else:
                depot_ids.append(depot_id)
        elif section == "done":
            raise ValueError(f"{path}: unexpected content after DEPOT_SECTION: {line}")
        else:
            raise AssertionError(f"unknown parser section {section!r}")

    required = {"NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "CAPACITY"}
    missing = required - headers.keys()
    if missing:
        raise ValueError(f"{path}: missing headers {sorted(missing)}")
    if headers["TYPE"].upper() != "CVRP":
        raise ValueError(f"{path}: TYPE must be CVRP")
    if headers["EDGE_WEIGHT_TYPE"].upper() != "EUC_2D":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE=EUC_2D is official for XML100")
    dimension = int(headers["DIMENSION"])
    if dimension != 101:
        raise ValueError(f"{path}: official XML100 dimension must be 101, got {dimension}")
    if len(coords_by_id) != dimension or len(demands_by_id) != dimension:
        raise ValueError(
            f"{path}: expected {dimension} coordinates/demands, got "
            f"{len(coords_by_id)}/{len(demands_by_id)}"
        )
    if set(coords_by_id) != set(demands_by_id):
        raise ValueError(f"{path}: coordinate and demand node IDs do not match")
    if set(coords_by_id) != set(range(1, dimension + 1)):
        raise ValueError(f"{path}: XML100 node IDs must be consecutive 1..{dimension}")
    if len(depot_ids) != 1:
        raise ValueError(f"{path}: expected exactly one depot, got {depot_ids}")
    depot_id = depot_ids[0]
    if depot_id != 1:
        raise ValueError(f"{path}: official XML100 depot node ID must be 1, got {depot_id}")
    if demands_by_id[depot_id] != 0:
        raise ValueError(f"{path}: depot demand must be zero")

    customer_node_ids = sorted(set(coords_by_id) - {depot_id})
    ordered_ids = [depot_id, *customer_node_ids]
    coords = np.asarray([coords_by_id[node] for node in ordered_ids], dtype=np.int64)
    demands = np.asarray([demands_by_id[node] for node in ordered_ids], dtype=np.int64)
    if (coords < 0).any() or (coords > 1000).any():
        raise ValueError(f"{path}: official XML100 coordinates must lie in [0, 1000]")
    name = headers["NAME"]
    if path.stem != name:
        raise ValueError(f"{path}: filename/name mismatch ({path.stem!r} != {name!r})")
    features = parse_xml100_name(name)
    capacity = int(headers["CAPACITY"])
    if capacity <= 0 or (demands[1:] <= 0).any() or (demands[1:] > capacity).any():
        raise ValueError(f"{path}: invalid capacity or customer demand")
    coords.setflags(write=False)
    demands.setflags(write=False)
    return CVRPInstance(name, capacity, coords, demands, path.resolve(), features)


def parse_solution(path: str | Path) -> ParsedSolution:
    path = Path(path)
    routes: list[list[int]] = []
    declared_cost: int | None = None
    route_numbers: list[int] = []
    route_re = re.compile(r"^Route\s+#(\d+)\s*:\s*(.*)$", re.IGNORECASE)
    cost_re = re.compile(r"^Cost\s*:?[ \t]*(-?\d+)\s*$", re.IGNORECASE)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        route_match = route_re.fullmatch(line)
        if route_match:
            route_numbers.append(int(route_match.group(1)))
            payload = route_match.group(2).strip()
            routes.append([int(token) for token in payload.split()] if payload else [])
            continue
        cost_match = cost_re.fullmatch(line)
        if cost_match:
            if declared_cost is not None:
                raise ValueError(f"{path}: duplicate Cost line")
            declared_cost = int(cost_match.group(1))
            continue
        raise ValueError(f"{path}: unsupported solution line: {line}")
    if not routes or declared_cost is None:
        raise ValueError(f"{path}: missing routes or Cost")
    if route_numbers != list(range(1, len(route_numbers) + 1)):
        raise ValueError(f"{path}: route numbers must be unique and consecutive from 1")
    if declared_cost <= 0:
        raise ValueError(f"{path}: solution cost must be positive")
    return ParsedSolution(routes, declared_cost, path.resolve())


def route_cost(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> int:
    matrix = instance.distance_matrix
    total = 0
    for route in routes:
        previous = 0
        for customer in route:
            total += int(matrix[previous, customer])
            previous = customer
        total += int(matrix[previous, 0])
    return total


def verify_routes(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> Verification:
    errors: list[str] = []
    expected = set(range(1, instance.customers + 1))
    seen: list[int] = []
    for route_index, route in enumerate(routes, start=1):
        if not route:
            errors.append(f"route {route_index} is empty")
            continue
        load = 0
        for customer in route:
            if not isinstance(customer, (int, np.integer)):
                errors.append(f"route {route_index} has non-integer customer {customer!r}")
                continue
            customer = int(customer)
            if customer not in expected:
                errors.append(f"route {route_index} has out-of-range customer {customer}")
                continue
            seen.append(customer)
            load += int(instance.demands[customer])
        if load > instance.capacity:
            errors.append(
                f"route {route_index} load {load} exceeds capacity {instance.capacity}"
            )
    counts = Counter(seen)
    seen_set = set(counts)
    missing = sorted(expected - seen_set)
    duplicates = sorted(customer for customer, count in counts.items() if count > 1)
    if missing:
        errors.append(f"missing customers: {missing}")
    if duplicates:
        errors.append(f"duplicate customers: {duplicates}")
    feasible = not errors
    cost = route_cost(instance, routes) if feasible else None
    return Verification(feasible, cost, len(routes), tuple(errors))


def actions_to_routes(actions: Iterable[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for raw in actions:
        node = int(raw)
        if node == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def split_giant_tour_dp(
    instance: CVRPInstance,
    giant_tours: np.ndarray | Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Bellman Split costs/predecessors under integer EUC_2D.

    Route reconstruction is deliberately separate so a large POMO pool need
    only reconstruct and verify the selected augmented/no-augmented candidates.
    """
    tours = np.asarray(giant_tours, dtype=np.int64)
    if tours.ndim == 1:
        tours = tours[None, :]
    count, n = tours.shape
    if n != instance.customers:
        raise ValueError(f"giant tour length {n} != customers {instance.customers}")
    expected = np.arange(1, n + 1, dtype=np.int64)
    if not np.all(np.sort(tours, axis=1) == expected[None, :]):
        raise ValueError("every giant tour must be a permutation of customer IDs 1..n")

    distance = instance.distance_matrix
    ordered_demand = instance.demands[tours]
    demand_prefix = np.concatenate(
        [np.zeros((count, 1), dtype=np.int64), np.cumsum(ordered_demand, axis=1)], axis=1
    )
    depot_edge = distance[0, tours]
    if n > 1:
        internal = distance[tours[:, :-1], tours[:, 1:]]
        edge_prefix = np.concatenate(
            [np.zeros((count, 1), dtype=np.int64), np.cumsum(internal, axis=1)], axis=1
        )
    else:
        edge_prefix = np.zeros((count, 1), dtype=np.int64)

    large = np.iinfo(np.int64).max // 8
    potential = np.full((count, n + 1), large, dtype=np.int64)
    predecessor = np.full((count, n + 1), -1, dtype=np.int16)
    potential[:, 0] = 0

    for end in range(1, n + 1):
        starts = np.arange(end)
        load = demand_prefix[:, [end]] - demand_prefix[:, :end]
        internal_cost = edge_prefix[:, [end - 1]] - edge_prefix[:, :end]
        segment_cost = depot_edge[:, :end] + internal_cost + depot_edge[:, [end - 1]]
        candidate = potential[:, :end] + segment_cost
        candidate[load > instance.capacity] = large
        best_start = np.argmin(candidate, axis=1)
        best_cost = candidate[np.arange(count), best_start]
        if np.any(best_cost >= large):
            raise ValueError("Split found no capacity-feasible partition")
        potential[:, end] = best_cost
        predecessor[:, end] = starts[best_start]

    return tours, potential[:, n].copy(), predecessor


def reconstruct_split_routes(
    instance: CVRPInstance,
    tours: np.ndarray,
    costs: np.ndarray,
    predecessor: np.ndarray,
    row_indices: Sequence[int] | None = None,
) -> list[list[list[int]]]:
    if row_indices is None:
        row_indices = range(len(tours))
    all_routes: list[list[list[int]]] = []
    n = instance.customers
    for raw_row in row_indices:
        row = int(raw_row)
        if row < 0 or row >= len(tours):
            raise IndexError(f"Split row index out of range: {row}")
        route_list: list[list[int]] = []
        end = n
        tour = tours[row].tolist()
        while end > 0:
            begin = int(predecessor[row, end])
            if begin < 0 or begin >= end:
                raise AssertionError("invalid Split predecessor chain")
            route_list.append(tour[begin:end])
            end = begin
        route_list.reverse()
        verification = verify_routes(instance, route_list)
        if not verification.feasible or verification.cost != int(costs[row]):
            raise AssertionError(
                f"Split reconstruction mismatch: {verification} vs dynamic cost {costs[row]}"
            )
        all_routes.append(route_list)
    return all_routes


def split_giant_tours_euc2d(
    instance: CVRPInstance,
    giant_tours: np.ndarray | Sequence[Sequence[int]],
) -> tuple[list[list[list[int]]], np.ndarray]:
    """Exact Bellman Split with full reconstruction (convenience/test API)."""
    tours, costs, predecessor = split_giant_tour_dp(instance, giant_tours)
    all_routes = reconstruct_split_routes(instance, tours, costs, predecessor)
    return all_routes, costs


def gap_percent(cost: int, optimum: int) -> float:
    if optimum <= 0:
        raise ValueError("optimum must be positive")
    return 100.0 * (float(cost) - float(optimum)) / float(optimum)
