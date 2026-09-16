from dataclasses import dataclass
from typing import Sequence

import torch

from .constraints import ConstraintSpec


@dataclass(frozen=True)
class VerificationResult:
    feasible: bool
    cost: float
    route_count: int
    reason: str = "ok"


def _edge_cost(a: torch.Tensor, b: torch.Tensor, scaler) -> float:
    value = float(torch.linalg.vector_norm(a.double() - b.double()).item())
    return round(value * scaler) / scaler if scaler is not None else value


def verify_routes(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    routes: Sequence[Sequence[int]],
    spec: ConstraintSpec,
    batch_index: int = 0,
    epsilon: float = 1e-6,
) -> VerificationResult:
    """Independent Python/double replay of one decoded solution."""
    spec.validate(node_xy)
    n = node_xy.size(1)
    flat = [int(customer) for route in routes for customer in route]
    if any(len(route) == 0 for route in routes):
        return VerificationResult(False, float("inf"), len(routes), "empty_route")
    if sorted(flat) != list(range(1, n + 1)):
        return VerificationResult(False, float("inf"), len(routes), "coverage_or_duplicate")

    depot = depot_xy[batch_index, 0]
    xy = node_xy[batch_index]
    demand = spec.demand[batch_index]
    capacity = float(spec.capacity.reshape(-1)[batch_index].item())
    total_cost = 0.0
    served = 0

    for route in routes:
        # Official VRPB semantics: a route starts full while any linehaul is
        # globally unserved, and starts empty once only backhauls remain.
        if spec.backhaul:
            suffix = flat[served:]
            has_delivery = any(float(demand[c - 1].item()) > 0 for c in suffix)
            load = capacity if has_delivery else 0.0
        else:
            load = capacity

        prev = depot
        raw_length = 0.0
        current_time = (
            float(spec.depot_start.reshape(-1)[batch_index].item())
            if spec.has_time_windows else 0.0
        )
        speed = float(spec.speed.reshape(-1)[batch_index].item()) if spec.has_time_windows else 1.0

        for customer in route:
            idx = customer - 1
            raw_edge = float(torch.linalg.vector_norm(prev.double() - xy[idx].double()).item())
            total_cost += _edge_cost(prev, xy[idx], spec.loc_scaler)
            raw_length += raw_edge
            load -= float(demand[idx].item())
            if load < -epsilon or load > capacity + epsilon:
                return VerificationResult(False, float("inf"), len(routes), "capacity")
            if spec.has_time_windows:
                arrival = max(
                    current_time + raw_edge / speed,
                    float(spec.tw_start[batch_index, idx].item()),
                )
                if arrival > float(spec.tw_end[batch_index, idx].item()) + epsilon:
                    return VerificationResult(False, float("inf"), len(routes), "customer_time_window")
                current_time = arrival + float(spec.service_time[batch_index, idx].item())
            prev = xy[idx]
            served += 1

        return_raw = float(torch.linalg.vector_norm(prev.double() - depot.double()).item())
        if not spec.open_route:
            total_cost += _edge_cost(prev, depot, spec.loc_scaler)
            raw_length += return_raw
        if spec.has_route_limit:
            limit = float(spec.route_limit.reshape(-1)[batch_index].item())
            if raw_length > limit + epsilon:
                return VerificationResult(False, float("inf"), len(routes), "route_limit")
        if spec.has_time_windows and not spec.open_route:
            depot_end = float(spec.depot_end.reshape(-1)[batch_index].item())
            if current_time + return_raw / speed > depot_end + epsilon:
                return VerificationResult(False, float("inf"), len(routes), "depot_return")

    return VerificationResult(True, total_cost, len(routes), "ok")
