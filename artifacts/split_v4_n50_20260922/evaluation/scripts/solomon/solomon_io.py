from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Sequence


NODE_RE = re.compile(r"^\s*(\d+)\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
VEHICLE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
ROUTE_RE = re.compile(r"^\s*Route\s*#?\s*\d+\s*:\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Node:
    customer: int
    x: int
    y: int
    demand: int
    ready: int
    due: int
    service: int


@dataclass(frozen=True)
class SolomonInstance:
    name: str
    max_vehicles: int
    capacity: int
    nodes: tuple[Node, ...]

    @property
    def customers(self) -> int:
        return len(self.nodes) - 1

    @property
    def depot(self) -> Node:
        return self.nodes[0]

    def subset(self, customers: int) -> "SolomonInstance":
        if not 1 <= customers <= self.customers:
            raise ValueError(f"invalid subset size {customers} for {self.name}")
        return SolomonInstance(self.name, self.max_vehicles, self.capacity, self.nodes[: customers + 1])

    def semantic_payload(self) -> dict:
        return {
            "name": self.name.upper(),
            "max_vehicles": self.max_vehicles,
            "capacity": self.capacity,
            "nodes": [asdict(node) for node in self.nodes],
        }

    def semantic_sha256(self) -> str:
        raw = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class Verification:
    feasible: bool
    vehicles: int
    distance_raw: float
    distance_2dp: str
    missing: list[int]
    duplicates: list[int]
    invalid_customers: list[int]
    capacity_violations: list[int]
    time_violations: list[str]
    fleet_violation: bool


def parse_instance(path: str | Path) -> SolomonInstance:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    name = next((line.strip() for line in lines if line.strip()), path.stem).upper()
    vehicle_index = next(i for i, line in enumerate(lines) if line.strip().upper() == "VEHICLE")
    max_vehicles = capacity = None
    for line in lines[vehicle_index + 1 :]:
        match = VEHICLE_RE.match(line)
        if match:
            max_vehicles, capacity = map(int, match.groups())
            break
    if max_vehicles is None:
        raise ValueError(f"cannot parse vehicle data from {path}")

    nodes = []
    for line in lines:
        match = NODE_RE.match(line)
        if match:
            nodes.append(Node(*map(int, match.groups())))
    if not nodes or nodes[0].customer != 0:
        raise ValueError(f"missing depot in {path}")
    expected = list(range(len(nodes)))
    actual = [node.customer for node in nodes]
    if actual != expected:
        raise ValueError(f"non-contiguous customer ids in {path}: {actual[:8]} ...")
    return SolomonInstance(name, max_vehicles, capacity, tuple(nodes))


def format_instance(instance: SolomonInstance) -> str:
    lines = [
        instance.name.upper(),
        "",
        "VEHICLE",
        "NUMBER     CAPACITY",
        f"{instance.max_vehicles:4d}{instance.capacity:12d}",
        "",
        "CUSTOMER",
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME",
        "",
    ]
    for n in instance.nodes:
        lines.append(
            f"{n.customer:5d}{n.x:11d}{n.y:11d}{n.demand:11d}"
            f"{n.ready:12d}{n.due:10d}{n.service:14d}"
        )
    return "\n".join(lines) + "\n"


def parse_solution(path: str | Path) -> list[list[int]]:
    routes = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="strict").splitlines():
        match = ROUTE_RE.match(line)
        if match:
            route = [int(token) for token in match.group(1).split() if int(token) != 0]
            if route:
                routes.append(route)
    if not routes:
        raise ValueError(f"no routes found in {path}")
    return routes


def actions_to_routes(actions: Iterable[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for value in actions:
        customer = int(value)
        if customer == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(customer)
    if current:
        routes.append(current)
    return routes


def verify_routes(
    instance: SolomonInstance,
    routes: Sequence[Sequence[int]],
    *,
    epsilon: float = 1e-9,
    enforce_fleet_limit: bool = True,
) -> Verification:
    n = instance.customers
    flat = [int(customer) for route in routes for customer in route]
    counts = {customer: flat.count(customer) for customer in set(flat)}
    invalid = sorted(customer for customer in counts if customer < 1 or customer > n)
    missing = sorted(set(range(1, n + 1)).difference(flat))
    duplicates = sorted(customer for customer, count in counts.items() if 1 <= customer <= n and count > 1)
    capacity_violations: list[int] = []
    time_violations: list[str] = []
    total = 0.0
    depot = instance.depot

    for route_index, route in enumerate(routes, start=1):
        load = 0
        time = float(depot.ready)
        previous = depot
        for customer in route:
            if not 1 <= int(customer) <= n:
                continue
            node = instance.nodes[int(customer)]
            travel = math.hypot(node.x - previous.x, node.y - previous.y)
            total += travel
            arrival = time + travel
            service_start = max(arrival, float(node.ready))
            if service_start > node.due + epsilon:
                time_violations.append(
                    f"route {route_index} customer {customer}: start={service_start:.12f} due={node.due}"
                )
            time = service_start + node.service
            load += node.demand
            previous = node
        total += math.hypot(previous.x - depot.x, previous.y - depot.y)
        return_time = time + math.hypot(previous.x - depot.x, previous.y - depot.y)
        if return_time > depot.due + epsilon:
            time_violations.append(
                f"route {route_index} depot return={return_time:.12f} due={depot.due}"
            )
        if load > instance.capacity:
            capacity_violations.append(route_index)

    fleet_violation = len(routes) > instance.max_vehicles
    feasible = not (
        invalid or missing or duplicates or capacity_violations or time_violations
        or (enforce_fleet_limit and fleet_violation)
    )
    rounded = Decimal.from_float(total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Verification(
        feasible=feasible,
        vehicles=len(routes),
        distance_raw=total,
        distance_2dp=f"{rounded:.2f}",
        missing=missing,
        duplicates=duplicates,
        invalid_customers=invalid,
        capacity_violations=capacity_violations,
        time_violations=time_violations,
        fleet_violation=fleet_violation,
    )


def split_giant_tour_double(
    instance: SolomonInstance,
    giant_tour: Sequence[int],
    *,
    epsilon: float = 1e-9,
    enforce_fleet_limit: bool = True,
) -> list[list[int]]:
    """Exact hard-capacity/hard-TW Split in original units and Python double.

    The customer permutation is fixed. Among all splits allowed by the selected
    fleet protocol, this returns the minimum-distance split. It is a
    deterministic decoder-side dynamic program, not a repair or local search.
    """
    tour = [int(customer) for customer in giant_tour]
    n = instance.customers
    if len(tour) != n or sorted(tour) != list(range(1, n + 1)):
        raise ValueError("giant_tour must be a permutation of customer ids 1..n")

    # segment_cost[start][end] is the depot-to-depot cost of tour[start:end].
    segment_cost: list[list[float | None]] = [[None] * (n + 1) for _ in range(n)]
    depot = instance.depot
    for start in range(n):
        load = 0
        clock = float(depot.ready)
        previous = depot
        travelled = 0.0
        for end in range(start, n):
            node = instance.nodes[tour[end]]
            load += node.demand
            if load > instance.capacity:
                break
            leg = math.hypot(node.x - previous.x, node.y - previous.y)
            travelled += leg
            service_start = max(clock + leg, float(node.ready))
            if service_start > node.due + epsilon:
                break
            clock = service_start + node.service
            previous = node
            back = math.hypot(previous.x - depot.x, previous.y - depot.y)
            if clock + back > depot.due + epsilon:
                break
            segment_cost[start][end + 1] = travelled + back

    # MVMoE Table 7 parses but does not enforce the Solomon fleet count.
    max_k = min(instance.max_vehicles, n) if enforce_fleet_limit else n
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(max_k + 1)]
    pred = [[-1] * (n + 1) for _ in range(max_k + 1)]
    best[0][0] = 0.0
    for vehicles in range(1, max_k + 1):
        for end in range(1, n + 1):
            for start in range(end):
                edge = segment_cost[start][end]
                if edge is None or not math.isfinite(best[vehicles - 1][start]):
                    continue
                candidate = best[vehicles - 1][start] + edge
                if candidate < best[vehicles][end] - 1e-12:
                    best[vehicles][end] = candidate
                    pred[vehicles][end] = start

    feasible_counts = [k for k in range(1, max_k + 1) if math.isfinite(best[k][n])]
    if not feasible_counts:
        raise ValueError("hard-TW Split found no feasible solution within fleet limit")
    vehicles = min(feasible_counts, key=lambda k: (best[k][n], k))
    routes: list[list[int]] = []
    end = n
    while end:
        start = pred[vehicles][end]
        if start < 0:
            raise RuntimeError("invalid exact Split predecessor chain")
        routes.append(tour[start:end])
        end = start
        vehicles -= 1
    routes.reverse()
    verification = verify_routes(
        instance,
        routes,
        epsilon=epsilon,
        enforce_fleet_limit=enforce_fleet_limit,
    )
    if not verification.feasible:
        raise RuntimeError(f"exact Split produced an infeasible result: {verification}")
    return routes


def normalized_tensors(instance: SolomonInstance, coordinate_scale: float = 100.0):
    import torch

    if coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive")
    depot, customers = instance.depot, instance.nodes[1:]
    depot_xy = torch.tensor([[[depot.x / coordinate_scale, depot.y / coordinate_scale]]], dtype=torch.float32)
    node_xy = torch.tensor(
        [[[node.x / coordinate_scale, node.y / coordinate_scale] for node in customers]],
        dtype=torch.float32,
    )
    demand = torch.tensor([[node.demand / instance.capacity for node in customers]], dtype=torch.float32)
    service = torch.tensor([[node.service / coordinate_scale for node in customers]], dtype=torch.float32)
    tw_start = torch.tensor([[node.ready / coordinate_scale for node in customers]], dtype=torch.float32)
    tw_end = torch.tensor([[node.due / coordinate_scale for node in customers]], dtype=torch.float32)
    return depot_xy, node_xy, demand, service, tw_start, tw_end
