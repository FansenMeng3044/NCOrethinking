"""Shared multi-vehicle CVRPTW semantics for AM and POMO experiments.

The environment represented here is the classical homogeneous-fleet CVRPTW:
each depot-delimited route is driven by a fresh, fully loaded vehicle whose
clock starts at ``depot_start``.  Routes are generated sequentially by the
neural decoder, but are independent in problem time and may execute in
parallel.  The objective is total Euclidean travel distance; time windows and
capacity are hard constraints rather than terms in the objective.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch


TensorOrFloat = Union[torch.Tensor, float]


@dataclass
class SplitResult:
    costs: torch.Tensor
    predecessors: Optional[torch.Tensor] = None
    route_counts: Optional[torch.Tensor] = None


@dataclass
class ReplayResult:
    distances: torch.Tensor
    route_counts: torch.Tensor
    all_customers_once: torch.Tensor
    capacity_feasible: torch.Tensor
    time_feasible: torch.Tensor
    feasible: torch.Tensor


def demand_scaler(problem_size: int) -> float:
    scalers = {20: 30.0, 50: 40.0, 100: 50.0, 200: 70.0}
    if problem_size not in scalers:
        raise NotImplementedError(
            "CVRPTW random generation supports sizes 20, 50, 100 and 200"
        )
    return scalers[problem_size]


def get_random_problems(
    batch_size: int,
    problem_size: int,
    *,
    device=None,
    depot_start: float = 0.0,
    depot_end: float = 3.0,
    speed: float = 1.0,
    service_duration: float = 0.2,
) -> Tuple[torch.Tensor, ...]:
    """Generate the same feasible-per-route distribution as ``VRPTWEnv.py``.

    Every customer is guaranteed to be feasible on its own route
    ``depot -> customer -> depot``.  This is sufficient for the unlimited
    homogeneous-fleet semantics because a fresh vehicle may serve each route.
    """

    if batch_size <= 0 or problem_size <= 0:
        raise ValueError("batch_size and problem_size must be positive")
    if speed <= 0 or depot_end <= depot_start or service_duration < 0:
        raise ValueError("invalid speed, depot interval, or service duration")

    depot_xy = torch.rand(batch_size, 1, 2, device=device)
    node_xy = torch.rand(batch_size, problem_size, 2, device=device)
    node_demand = torch.randint(
        1, 10, (batch_size, problem_size), device=device
    ).float() / demand_scaler(problem_size)
    service_time = torch.full(
        (batch_size, problem_size),
        float(service_duration),
        device=device,
        dtype=node_xy.dtype,
    )

    travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / speed
    earliest_center = depot_start + travel_time
    latest_center = depot_end - travel_time - service_time
    if (latest_center < earliest_center).any():
        # With the supported defaults this cannot happen.  Keep the failure
        # explicit for custom horizons instead of recursing indefinitely.
        raise ValueError(
            "depot horizon is too short for the requested service duration"
        )
    centers = earliest_center + torch.rand_like(travel_time) * (
        latest_center - earliest_center
    )
    max_half_width = (depot_end - depot_start) / 3.0
    min_half_width = service_time / 2.0
    half_width = min_half_width + torch.rand_like(travel_time) * (
        max_half_width - min_half_width
    )
    tw_start = torch.clamp(centers - half_width, min=depot_start, max=depot_end)
    tw_end = torch.clamp(centers + half_width, min=depot_start, max=depot_end)

    return depot_xy, node_xy, node_demand, service_time, tw_start, tw_end


def augment_xy_by_8(xy: torch.Tensor) -> torch.Tensor:
    x = xy[..., [0]]
    y = xy[..., [1]]
    return torch.cat(
        (
            torch.cat((x, y), dim=-1),
            torch.cat((1 - x, y), dim=-1),
            torch.cat((x, 1 - y), dim=-1),
            torch.cat((1 - x, 1 - y), dim=-1),
            torch.cat((y, x), dim=-1),
            torch.cat((1 - y, x), dim=-1),
            torch.cat((y, 1 - x), dim=-1),
            torch.cat((1 - y, 1 - x), dim=-1),
        ),
        dim=0,
    )


def augment_problems_by_8(problems: Tuple[torch.Tensor, ...]):
    depot_xy, node_xy, demand, service, tw_start, tw_end = problems
    return (
        augment_xy_by_8(depot_xy),
        augment_xy_by_8(node_xy),
        demand.repeat(8, 1),
        service.repeat(8, 1),
        tw_start.repeat(8, 1),
        tw_end.repeat(8, 1),
    )


def tensors_from_dict(data: dict, *, device=None) -> Tuple[torch.Tensor, ...]:
    """Read the canonical fixed-test-set tensor dictionary."""

    aliases = {
        "depot_xy": ("depot_xy", "depot"),
        "node_xy": ("node_xy", "loc"),
        "node_demand": ("node_demand", "demand"),
        "service_time": ("node_service_time", "service_time"),
        "tw_start": ("node_tw_start", "tw_start"),
        "tw_end": ("node_tw_end", "tw_end"),
    }
    values = []
    for label, names in aliases.items():
        value = next((data[name] for name in names if name in data), None)
        if value is None:
            raise KeyError("CVRPTW dataset is missing {} ({})".format(label, names))
        values.append(torch.as_tensor(value, dtype=torch.float, device=device))
    depot = values[0]
    if depot.dim() == 2:
        depot = depot[:, None, :]
    values[0] = depot
    validate_problem_tensors(*values)
    return tuple(values)


def validate_problem_tensors(
    depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
):
    if depot_xy.dim() != 3 or depot_xy.shape[1:] != (1, 2):
        raise ValueError("depot_xy must have shape (batch, 1, 2)")
    if node_xy.dim() != 3 or node_xy.size(2) != 2:
        raise ValueError("node_xy must have shape (batch, n, 2)")
    expected = node_xy.shape[:2]
    for name, tensor in (
        ("node_demand", node_demand),
        ("service_time", service_time),
        ("tw_start", tw_start),
        ("tw_end", tw_end),
    ):
        if tensor.shape != expected:
            raise ValueError("{} must have shape {}".format(name, tuple(expected)))
    if (node_demand < 0).any() or (service_time < 0).any():
        raise ValueError("demands and service times must be non-negative")
    if (tw_end < tw_start).any():
        raise ValueError("every time window must satisfy tw_start <= tw_end")


def split_giant_tours_tw(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    service_time: torch.Tensor,
    tw_start: torch.Tensor,
    tw_end: torch.Tensor,
    giant_tours: torch.Tensor,
    *,
    capacity: float = 1.0,
    depot_start: TensorOrFloat = 0.0,
    depot_end: TensorOrFloat = 3.0,
    speed: float = 1.0,
    return_predecessors: bool = False,
    epsilon: float = 1e-6,
) -> SplitResult:
    """Optimal hard-capacity/hard-TW Split for a fixed customer order.

    The customer permutation is not changed.  Each auxiliary edge represents
    one independent vehicle route that starts with full capacity and time
    ``depot_start``.  The shortest path minimises total travel distance.
    """

    validate_problem_tensors(
        depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
    )
    if capacity <= 0 or speed <= 0:
        raise ValueError("capacity and speed must be positive")
    squeeze_pomo = giant_tours.dim() == 2
    if squeeze_pomo:
        giant_tours = giant_tours[:, None, :]
    if giant_tours.dim() != 3:
        raise ValueError("giant_tours must have shape (batch, n) or (batch, pomo, n)")

    batch_size, problem_size, _ = node_xy.shape
    pomo_size = giant_tours.size(1)
    if giant_tours.shape != (batch_size, pomo_size, problem_size):
        raise ValueError("giant_tours shape does not match problem tensors")
    expected = torch.arange(
        1, problem_size + 1, device=giant_tours.device, dtype=giant_tours.dtype
    )
    if not torch.equal(
        giant_tours.sort(dim=2).values,
        expected[None, None, :].expand_as(giant_tours),
    ):
        raise ValueError("each giant tour must be a permutation of customer ids 1..n")

    route_count = batch_size * pomo_size
    gather_xy = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, -1, 2)
    ordered_xy = (
        node_xy[:, None].expand(-1, pomo_size, -1, -1)
        .gather(2, gather_xy)
        .reshape(route_count, problem_size, 2)
    )

    def gather_feature(feature):
        return (
            feature[:, None].expand(-1, pomo_size, -1)
            .gather(2, giant_tours - 1)
            .reshape(route_count, problem_size)
        )

    ordered_demand = gather_feature(node_demand)
    ordered_service = gather_feature(service_time)
    ordered_tw_start = gather_feature(tw_start)
    ordered_tw_end = gather_feature(tw_end)
    depot = (
        depot_xy[:, None].expand(-1, pomo_size, -1, -1)
        .reshape(route_count, 1, 2)
    )
    depot_distance = (ordered_xy - depot).norm(p=2, dim=-1)
    consecutive = (
        (ordered_xy[:, 1:] - ordered_xy[:, :-1]).norm(p=2, dim=-1)
        if problem_size > 1
        else ordered_xy.new_zeros(route_count, 0)
    )
    edge_prefix = torch.cat(
        (ordered_xy.new_zeros(route_count, 1), consecutive.cumsum(dim=1)), dim=1
    )
    demand_prefix = torch.cat(
        (ordered_demand.new_zeros(route_count, 1), ordered_demand.cumsum(dim=1)),
        dim=1,
    )

    route_depot_start = _expand_batch_scalar(
        depot_start, batch_size, pomo_size, node_xy
    )
    route_depot_end = _expand_batch_scalar(depot_end, batch_size, pomo_size, node_xy)
    inf = torch.tensor(float("inf"), device=node_xy.device, dtype=node_xy.dtype)
    potential = torch.full(
        (route_count, problem_size + 1), inf, device=node_xy.device, dtype=node_xy.dtype
    )
    potential[:, 0] = 0
    route_counts = torch.full(
        (route_count, problem_size + 1), problem_size + 1,
        device=node_xy.device, dtype=torch.long,
    )
    route_counts[:, 0] = 0
    predecessors = torch.full(
        (route_count, problem_size + 1), -1,
        device=node_xy.device, dtype=torch.long,
    ) if return_predecessors else None

    # completion[:, start] is completion time at the current `end` customer
    # for the segment ordered[start:end+1].
    completion = torch.zeros(
        route_count, problem_size, device=node_xy.device, dtype=node_xy.dtype
    )
    time_valid = torch.ones(
        route_count, problem_size, device=node_xy.device, dtype=torch.bool
    )

    for end in range(problem_size):
        if end > 0:
            arrival = completion[:, :end] + consecutive[:, end - 1, None] / speed
            service_start = torch.maximum(arrival, ordered_tw_start[:, end, None])
            completion[:, :end] = service_start + ordered_service[:, end, None]
            time_valid[:, :end] &= (
                service_start <= ordered_tw_end[:, end, None] + epsilon
            )

        first_arrival = route_depot_start[:, 0] + depot_distance[:, end] / speed
        first_service_start = torch.maximum(first_arrival, ordered_tw_start[:, end])
        completion[:, end] = first_service_start + ordered_service[:, end]
        time_valid[:, end] = first_service_start <= ordered_tw_end[:, end] + epsilon

        starts = slice(0, end + 1)
        segment_load = demand_prefix[:, end + 1, None] - demand_prefix[:, : end + 1]
        internal_distance = edge_prefix[:, end, None] - edge_prefix[:, : end + 1]
        segment_cost = (
            depot_distance[:, : end + 1]
            + internal_distance
            + depot_distance[:, end, None]
        )
        can_return = (
            completion[:, starts] + depot_distance[:, end, None] / speed
            <= route_depot_end + epsilon
        )
        feasible = (
            time_valid[:, starts]
            & can_return
            & (segment_load <= capacity + epsilon)
        )
        candidate = (potential[:, : end + 1] + segment_cost).masked_fill(
            ~feasible, inf
        )
        best_cost, best_start = candidate.min(dim=1)
        potential[:, end + 1] = best_cost
        route_counts[:, end + 1] = route_counts.gather(
            1, best_start[:, None]
        ).squeeze(1) + 1
        if predecessors is not None:
            predecessors[:, end + 1] = best_start

    final_cost = potential[:, -1]
    if not torch.isfinite(final_cost).all():
        raise ValueError(
            "hard-TW Split found no feasible solution; check capacity, time windows, "
            "service times, speed, and depot horizon"
        )
    output_shape = (batch_size, pomo_size)
    costs = final_cost.reshape(output_shape)
    counts = route_counts[:, -1].reshape(output_shape)
    pred_out = None
    if predecessors is not None:
        pred_out = predecessors.reshape(batch_size, pomo_size, problem_size + 1)
    if squeeze_pomo:
        costs = costs[:, 0]
        counts = counts[:, 0]
        if pred_out is not None:
            pred_out = pred_out[:, 0]
    return SplitResult(costs, pred_out, counts)


def reconstruct_routes(
    giant_tour: torch.Tensor, predecessors: torch.Tensor
) -> List[List[int]]:
    if giant_tour.dim() != 1 or predecessors.dim() != 1:
        raise ValueError("giant_tour and predecessors must be rank-1 tensors")
    if predecessors.numel() != giant_tour.numel() + 1:
        raise ValueError("predecessors must contain n+1 entries")
    tour = giant_tour.detach().cpu().tolist()
    pred = predecessors.detach().cpu().tolist()
    routes = []
    end = len(tour)
    while end:
        start = pred[end]
        if start < 0 or start >= end:
            raise ValueError("invalid Split predecessor chain")
        routes.append(tour[start:end])
        end = start
    routes.reverse()
    return routes


def replay_cvrptw_actions(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    service_time: torch.Tensor,
    tw_start: torch.Tensor,
    tw_end: torch.Tensor,
    actions: torch.Tensor,
    *,
    capacity: float = 1.0,
    depot_start: TensorOrFloat = 0.0,
    depot_end: TensorOrFloat = 3.0,
    speed: float = 1.0,
    epsilon: float = 1e-6,
) -> ReplayResult:
    """Strictly replay depot-delimited routes and return per-instance checks."""

    validate_problem_tensors(
        depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
    )
    if actions.dim() != 2 or actions.size(0) != node_xy.size(0):
        raise ValueError("actions must have shape (batch, sequence_length)")
    batch_size, problem_size, _ = node_xy.shape
    if ((actions < 0) | (actions > problem_size)).any():
        raise ValueError("actions contain an out-of-range node id")

    device, dtype = node_xy.device, node_xy.dtype
    distances = torch.zeros(batch_size, device=device, dtype=dtype)
    route_counts = torch.zeros(batch_size, device=device, dtype=torch.long)
    visit_count = torch.zeros(batch_size, problem_size, device=device, dtype=torch.long)
    capacity_ok = torch.ones(batch_size, device=device, dtype=torch.bool)
    time_ok = torch.ones(batch_size, device=device, dtype=torch.bool)
    load = torch.zeros(batch_size, device=device, dtype=dtype)
    current_time = _expand_batch_scalar(
        depot_start, batch_size, 1, node_xy
    )[:, 0].clone()
    horizon = _expand_batch_scalar(depot_end, batch_size, 1, node_xy)[:, 0]
    current_coord = depot_xy[:, 0]
    route_open = torch.zeros(batch_size, device=device, dtype=torch.bool)
    batch_index = torch.arange(batch_size, device=device)

    for column in range(actions.size(1)):
        selected = actions[:, column]
        is_depot = selected == 0
        customer_index = (selected - 1).clamp(min=0)
        selected_coord = torch.where(
            is_depot[:, None],
            depot_xy[:, 0],
            node_xy[batch_index, customer_index],
        )
        travel = (selected_coord - current_coord).norm(p=2, dim=-1)
        distances += travel

        if (~is_depot).any():
            active = ~is_depot
            idx = customer_index[active]
            rows = batch_index[active]
            new_route = active & ~route_open
            route_counts += new_route.long()
            route_open |= active
            load[active] += node_demand[rows, idx]
            capacity_ok[active] &= load[active] <= capacity + epsilon
            service_start = torch.maximum(
                current_time[active] + travel[active] / speed,
                tw_start[rows, idx],
            )
            time_ok[active] &= service_start <= tw_end[rows, idx] + epsilon
            current_time[active] = service_start + service_time[rows, idx]
            visit_count[rows, idx] += 1

        if is_depot.any():
            current_time[is_depot] += travel[is_depot] / speed
            time_ok[is_depot] &= current_time[is_depot] <= horizon[is_depot] + epsilon
            load[is_depot] = 0
            current_time[is_depot] = _expand_batch_scalar(
                depot_start, batch_size, 1, node_xy
            )[is_depot, 0]
            route_open[is_depot] = False
        current_coord = selected_coord

    # AM may omit the final depot action; close the last route implicitly.
    return_distance = (current_coord - depot_xy[:, 0]).norm(p=2, dim=-1)
    distances += return_distance
    time_ok &= current_time + return_distance / speed <= horizon + epsilon
    all_once = (visit_count == 1).all(dim=1)
    feasible = all_once & capacity_ok & time_ok
    return ReplayResult(
        distances, route_counts, all_once, capacity_ok, time_ok, feasible
    )


def _expand_batch_scalar(value, batch_size, repeat, reference):
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(batch_size * repeat, 1)
    tensor = tensor.reshape(batch_size, -1)
    if tensor.size(1) != 1:
        raise ValueError("depot time tensors must be scalar or have one value per batch")
    return tensor[:, None, :].expand(-1, repeat, -1).reshape(batch_size * repeat, 1)
