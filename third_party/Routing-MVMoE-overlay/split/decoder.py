from dataclasses import dataclass
from typing import List, Optional

import torch

from .constraints import ConstraintSpec


@dataclass
class SplitResult:
    costs: torch.Tensor
    feasible: torch.Tensor
    route_counts: torch.Tensor
    predecessors: Optional[torch.Tensor] = None


def _as_batch_vector(value: torch.Tensor, batch: int, dtype, device) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).reshape(batch)


def _rounded_distance(distance: torch.Tensor, loc_scaler: Optional[float]) -> torch.Tensor:
    if loc_scaler is None:
        return distance
    if loc_scaler <= 0:
        raise ValueError("loc_scaler must be positive")
    return torch.round(distance * loc_scaler) / loc_scaler


def _validate_tours(depot_xy, node_xy, giant_tours, spec):
    if depot_xy.ndim != 3 or depot_xy.shape[1:] != (1, 2):
        raise ValueError("depot_xy must have shape (batch, 1, 2)")
    if node_xy.ndim != 3 or node_xy.size(-1) != 2:
        raise ValueError("node_xy must have shape (batch, n, 2)")
    if giant_tours.ndim != 3:
        raise ValueError("giant_tours must have shape (batch, pomo, n)")
    batch, n, _ = node_xy.shape
    if depot_xy.size(0) != batch or giant_tours.size(0) != batch or giant_tours.size(2) != n:
        raise ValueError("batch/problem dimensions do not agree")
    if not torch.isfinite(depot_xy).all() or not torch.isfinite(node_xy).all():
        raise ValueError("coordinates contain a non-finite value")
    expected = torch.arange(1, n + 1, device=giant_tours.device, dtype=giant_tours.dtype)
    if not torch.equal(giant_tours.sort(dim=2).values, expected.view(1, 1, n).expand_as(giant_tours)):
        raise ValueError("every giant tour must be a permutation of customer IDs 1..n")
    spec.validate(node_xy)


def _validate_mandatory_breaks(giant_tours, mandatory_breaks):
    if mandatory_breaks is None:
        mandatory_breaks = torch.zeros_like(giant_tours, dtype=torch.bool)
        mandatory_breaks[:, :, 0] = True
        return mandatory_breaks
    if mandatory_breaks.shape != giant_tours.shape:
        raise ValueError("mandatory_breaks must have the same shape as giant_tours")
    if mandatory_breaks.dtype != torch.bool:
        raise ValueError("mandatory_breaks must be boolean")
    if not mandatory_breaks[:, :, 0].all():
        raise ValueError("the first customer must start a mandatory B segment")
    return mandatory_breaks.to(device=giant_tours.device)


def split_giant_tours(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    giant_tours: torch.Tensor,
    spec: ConstraintSpec,
    mandatory_breaks: Optional[torch.Tensor] = None,
    return_predecessors: bool = False,
    epsilon: float = 1e-6,
) -> SplitResult:
    """Exact O(n^2) B/L/C/TW Split over a fixed customer order.

    The implementation is vectorized over batch, POMO trajectories, and all
    candidate route starts.  Only the endpoint loop is sequential.  Constraint
    feasibility uses raw Euclidean distances, exactly like the official envs.
    B is exposed during order decoding, but Split independently rechecks signed
    B load semantics for every candidate segment and is free to choose better
    B boundaries. L is not exposed during decoding and is enforced only here.
    ``mandatory_breaks`` is an optional diagnostic restriction, not part of the
    training pipeline. Optional ``loc_scaler`` rounding is applied only to
    objective edges.
    """
    _validate_tours(depot_xy, node_xy, giant_tours, spec)
    mandatory_breaks = _validate_mandatory_breaks(giant_tours, mandatory_breaks)
    batch, n, _ = node_xy.shape
    pomo = giant_tours.size(1)
    rows = batch * pomo
    device, dtype = node_xy.device, node_xy.dtype

    xy_idx = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, -1, 2)
    ordered_xy = node_xy[:, None].expand(-1, pomo, -1, -1).gather(2, xy_idx).reshape(rows, n, 2)
    demand_idx = giant_tours - 1
    ordered_demand = spec.demand[:, None].expand(-1, pomo, -1).gather(2, demand_idx).reshape(rows, n)
    mandatory_breaks = mandatory_breaks.reshape(rows, n)
    positions = torch.arange(n, device=device)[None, :].expand(rows, -1)
    mandatory_positions = torch.where(
        mandatory_breaks, positions, torch.zeros_like(positions)
    )
    last_mandatory_start = torch.cummax(mandatory_positions, dim=1).values
    depot = depot_xy[:, None].expand(-1, pomo, -1, -1).reshape(rows, 1, 2)

    depot_dist_raw = (ordered_xy - depot).norm(p=2, dim=-1)
    if n > 1:
        between_raw = (ordered_xy[:, 1:] - ordered_xy[:, :-1]).norm(p=2, dim=-1)
    else:
        between_raw = ordered_xy.new_zeros((rows, 0))
    depot_dist_cost = _rounded_distance(depot_dist_raw, spec.loc_scaler)
    between_cost = _rounded_distance(between_raw, spec.loc_scaler)

    capacity = _as_batch_vector(spec.capacity, batch, dtype, device)
    capacity = capacity[:, None].expand(-1, pomo).reshape(rows)
    if spec.backhaul:
        suffix_has_delivery = torch.flip(
            torch.cumsum(torch.flip((ordered_demand > 0).to(torch.int64), dims=(1,)), dim=1),
            dims=(1,),
        ) > 0
        initial_load = torch.where(
            suffix_has_delivery,
            capacity[:, None].expand(-1, n),
            torch.zeros(rows, n, device=device, dtype=dtype),
        )
    else:
        initial_load = capacity[:, None].expand(-1, n).clone()

    load = initial_load.clone()
    path_raw = torch.zeros(rows, n, device=device, dtype=dtype)
    path_cost = torch.zeros_like(path_raw)
    valid = torch.zeros(rows, n, device=device, dtype=torch.bool)

    if spec.has_time_windows:
        ordered_service = spec.service_time[:, None].expand(-1, pomo, -1).gather(2, demand_idx).reshape(rows, n)
        ordered_tw_start = spec.tw_start[:, None].expand(-1, pomo, -1).gather(2, demand_idx).reshape(rows, n)
        ordered_tw_end = spec.tw_end[:, None].expand(-1, pomo, -1).gather(2, demand_idx).reshape(rows, n)
        depot_start = _as_batch_vector(spec.depot_start, batch, dtype, device)[:, None].expand(-1, pomo).reshape(rows)
        depot_end = _as_batch_vector(spec.depot_end, batch, dtype, device)[:, None].expand(-1, pomo).reshape(rows)
        speed = _as_batch_vector(spec.speed, batch, dtype, device)[:, None].expand(-1, pomo).reshape(rows)
        completion = torch.zeros(rows, n, device=device, dtype=dtype)
    else:
        ordered_service = ordered_tw_start = ordered_tw_end = None
        depot_start = depot_end = speed = completion = None

    if spec.has_route_limit:
        route_limit = _as_batch_vector(spec.route_limit, batch, dtype, device)
        route_limit = route_limit[:, None].expand(-1, pomo).reshape(rows)
    else:
        route_limit = None

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    potential = torch.full((rows, n + 1), inf, device=device, dtype=dtype)
    potential[:, 0] = 0
    route_labels = torch.full((rows, n + 1), n + 1, device=device, dtype=torch.long)
    route_labels[:, 0] = 0
    predecessors = None
    if return_predecessors:
        predecessors = torch.full((rows, n + 1), -1, device=device, dtype=torch.long)

    row_idx = torch.arange(rows, device=device)
    for end in range(n):
        active = end + 1
        valid[:, end] = True

        # Update route resources for every candidate start <= end.
        load[:, :active] -= ordered_demand[:, end, None]
        valid[:, :active] &= load[:, :active] >= -epsilon
        valid[:, :active] &= load[:, :active] <= capacity[:, None] + epsilon
        starts = torch.arange(active, device=device)[None, :]
        valid[:, :active] &= starts >= last_mandatory_start[:, end, None]

        if end == 0:
            path_raw[:, 0] = depot_dist_raw[:, 0]
            path_cost[:, 0] = depot_dist_cost[:, 0]
        else:
            path_raw[:, :end] += between_raw[:, end - 1, None]
            path_cost[:, :end] += between_cost[:, end - 1, None]
            path_raw[:, end] = depot_dist_raw[:, end]
            path_cost[:, end] = depot_dist_cost[:, end]

        if spec.has_route_limit:
            route_length = path_raw[:, :active]
            if not spec.open_route:
                route_length = route_length + depot_dist_raw[:, end, None]
            valid[:, :active] &= route_length <= route_limit[:, None] + epsilon

        if spec.has_time_windows:
            if end > 0:
                arrivals_old = torch.maximum(
                    completion[:, :end] + between_raw[:, end - 1, None] / speed[:, None],
                    ordered_tw_start[:, end, None],
                )
                completion[:, :end] = arrivals_old + ordered_service[:, end, None]
                valid[:, :end] &= arrivals_old <= ordered_tw_end[:, end, None] + epsilon
            arrival_new = torch.maximum(
                depot_start + depot_dist_raw[:, end] / speed,
                ordered_tw_start[:, end],
            )
            completion[:, end] = arrival_new + ordered_service[:, end]
            valid[:, end] &= arrival_new <= ordered_tw_end[:, end] + epsilon
            if not spec.open_route:
                valid[:, :active] &= (
                    completion[:, :active] + depot_dist_raw[:, end, None] / speed[:, None]
                    <= depot_end[:, None] + epsilon
                )

        edge_cost = path_cost[:, :active]
        if not spec.open_route:
            edge_cost = edge_cost + depot_dist_cost[:, end, None]
        candidate = potential[:, :active] + edge_cost
        candidate = candidate.masked_fill(~valid[:, :active], inf)
        candidate_routes = route_labels[:, :active] + 1

        best_cost, best_start = candidate.min(dim=1)
        # Deterministic secondary tie-break: fewer routes, then earliest start.
        tied = torch.isclose(candidate, best_cost[:, None], rtol=0.0, atol=1e-9)
        tied_routes = candidate_routes.masked_fill(~tied, n + 1)
        best_route_count, tie_start = tied_routes.min(dim=1)
        use_tie = torch.isfinite(best_cost)
        best_start = torch.where(use_tie, tie_start, best_start)
        potential[:, end + 1] = best_cost
        route_labels[:, end + 1] = torch.where(
            use_tie, best_route_count, torch.full_like(best_route_count, n + 1)
        )
        if predecessors is not None:
            predecessors[:, end + 1] = torch.where(
                use_tie, best_start, torch.full_like(best_start, -1)
            )

    costs = potential[:, n].reshape(batch, pomo)
    feasible = torch.isfinite(costs)
    counts = route_labels[:, n].reshape(batch, pomo)
    counts = torch.where(feasible, counts, torch.full_like(counts, -1))
    pred_out = predecessors.reshape(batch, pomo, n + 1) if predecessors is not None else None
    return SplitResult(costs=costs, feasible=feasible, route_counts=counts, predecessors=pred_out)


def reconstruct_routes(giant_tour: torch.Tensor, predecessors: torch.Tensor) -> List[List[int]]:
    if giant_tour.ndim != 1 or predecessors.ndim != 1:
        raise ValueError("giant_tour and predecessors must be rank-1")
    if predecessors.numel() != giant_tour.numel() + 1:
        raise ValueError("predecessors must contain n+1 entries")
    tour = giant_tour.detach().cpu().tolist()
    pred = predecessors.detach().cpu().tolist()
    routes = []
    end = len(tour)
    while end:
        start = pred[end]
        if start < 0 or start >= end:
            raise ValueError("cannot reconstruct an infeasible or corrupt Split result")
        routes.append(tour[start:end])
        end = start
    routes.reverse()
    return routes
