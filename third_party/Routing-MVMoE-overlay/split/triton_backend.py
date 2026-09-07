"""Exact Triton backend for the fixed-order B/L/C/TW Split dynamic program.

The public reference implementation in :mod:`split.decoder` remains the
correctness oracle.  This backend changes only execution: it fuses the
per-endpoint tensor updates into one segment kernel and the Bellman recurrence
into one dynamic-programming kernel per flattened batch/POMO row.
"""

from typing import Optional

import torch

from .constraints import FEASIBILITY_EPSILON, ConstraintSpec
from .decoder import SplitResult, _as_batch_vector, _rounded_distance


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - exercised on CUDA hosts
        raise RuntimeError(
            "The Triton Split backend requires a PyTorch installation with Triton"
        ) from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _segment_edges_kernel(
    ordered_demand,
    initial_load,
    last_mandatory_start,
    depot_dist_raw,
    between_raw,
    depot_dist_cost,
    between_cost,
    capacity,
    route_limit,
    ordered_service,
    ordered_tw_start,
    ordered_tw_end,
    depot_start,
    depot_end,
    speed,
    edge_costs,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ROUTE_LIMIT: tl.constexpr,
    HAS_TIME_WINDOWS: tl.constexpr,
    OPEN_ROUTE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    starts = tl.arange(0, BLOCK)
    in_problem = starts < N
    row_offset = row * N

    load = tl.load(initial_load + row_offset + starts, mask=in_problem, other=0.0)
    path_raw = tl.zeros((BLOCK,), tl.float32)
    path_cost = tl.zeros((BLOCK,), tl.float32)
    completion = tl.zeros((BLOCK,), tl.float32)
    valid = tl.zeros((BLOCK,), tl.int1)

    cap = tl.load(capacity + row)
    if HAS_ROUTE_LIMIT:
        limit = tl.load(route_limit + row)
    if HAS_TIME_WINDOWS:
        route_depot_start = tl.load(depot_start + row)
        route_depot_end = tl.load(depot_end + row)
        route_speed = tl.load(speed + row)

    for end in range(0, N):
        active = in_problem & (starts <= end)
        new_start = in_problem & (starts == end)
        old_start = in_problem & (starts < end)

        demand = tl.load(ordered_demand + row_offset + end)
        load = tl.where(active, load - demand, load)
        valid = tl.where(new_start, True, valid)
        feasible = (load >= -EPSILON) & (load <= cap + EPSILON)
        mandatory_start = tl.load(last_mandatory_start + row_offset + end)
        feasible = feasible & (starts >= mandatory_start)
        valid = tl.where(active, valid & feasible, valid)

        depot_raw = tl.load(depot_dist_raw + row_offset + end)
        depot_cost = tl.load(depot_dist_cost + row_offset + end)
        between_raw_value = 0.0
        between_cost_value = 0.0
        if end == 0:
            path_raw = tl.where(new_start, depot_raw, path_raw)
            path_cost = tl.where(new_start, depot_cost, path_cost)
        else:
            between_raw_value = tl.load(between_raw + row * (N - 1) + end - 1)
            between_cost_value = tl.load(between_cost + row * (N - 1) + end - 1)
            path_raw = tl.where(
                new_start,
                depot_raw,
                tl.where(old_start, path_raw + between_raw_value, path_raw),
            )
            path_cost = tl.where(
                new_start,
                depot_cost,
                tl.where(old_start, path_cost + between_cost_value, path_cost),
            )

        if HAS_ROUTE_LIMIT:
            route_length = path_raw
            if not OPEN_ROUTE:
                route_length = route_length + depot_raw
            valid = tl.where(
                active, valid & (route_length <= limit + EPSILON), valid
            )

        if HAS_TIME_WINDOWS:
            service = tl.load(ordered_service + row_offset + end)
            window_start = tl.load(ordered_tw_start + row_offset + end)
            window_end = tl.load(ordered_tw_end + row_offset + end)
            arrival_new = tl.maximum(
                route_depot_start + depot_raw / route_speed, window_start
            )
            arrival_old = tl.maximum(
                completion + between_raw_value / route_speed, window_start
            )
            arrival = tl.where(new_start, arrival_new, arrival_old)
            completion = tl.where(active, arrival + service, completion)
            valid = tl.where(active, valid & (arrival <= window_end + EPSILON), valid)
            if not OPEN_ROUTE:
                valid = tl.where(
                    active,
                    valid
                    & (
                        completion + depot_raw / route_speed
                        <= route_depot_end + EPSILON
                    ),
                    valid,
                )

        edge = path_cost
        if not OPEN_ROUTE:
            edge = edge + depot_cost
        output_offset = row * N * N + starts * N + end
        tl.store(
            edge_costs + output_offset,
            tl.where(valid, edge, float("inf")),
            mask=active,
        )


@triton.jit
def _split_dp_kernel(
    edge_costs,
    final_costs,
    final_route_counts,
    predecessors,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    vertices = tl.arange(0, BLOCK)
    in_vertices = vertices <= N
    potential = tl.where(vertices == 0, 0.0, float("inf"))
    route_labels = tl.where(vertices == 0, 0, N + 1)

    for end in range(0, N):
        active = vertices <= end
        edge_offset = row * N * N + vertices * N + end
        edge = tl.load(edge_costs + edge_offset, mask=active, other=float("inf"))
        candidate = tl.where(active, potential + edge, float("inf"))
        best_cost = tl.min(candidate, axis=0)
        finite = best_cost < float("inf")
        tied = active & finite & (tl.abs(candidate - best_cost) <= 1.0e-9)
        candidate_routes = route_labels + 1
        best_routes = tl.min(
            tl.where(tied, candidate_routes, N + 1), axis=0
        )
        tied_best_routes = tied & (candidate_routes == best_routes)
        best_start = tl.min(
            tl.where(tied_best_routes, vertices, N + 1), axis=0
        )
        destination = vertices == end + 1
        potential = tl.where(
            destination, tl.where(finite, best_cost, float("inf")), potential
        )
        route_labels = tl.where(
            destination, tl.where(finite, best_routes, N + 1), route_labels
        )
        tl.store(
            predecessors + row * (N + 1) + end + 1,
            tl.where(finite, best_start, -1),
        )

    final_mask = in_vertices & (vertices == N)
    cost = tl.sum(tl.where(final_mask, potential, 0.0), axis=0)
    routes = tl.sum(tl.where(final_mask, route_labels, 0), axis=0)
    feasible = cost < float("inf")
    tl.store(final_costs + row, cost)
    tl.store(final_route_counts + row, tl.where(feasible, routes, -1))
    tl.store(predecessors + row * (N + 1), -1)


def split_giant_tours_triton(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    giant_tours: torch.Tensor,
    spec: ConstraintSpec,
    mandatory_breaks: torch.Tensor,
    return_predecessors: bool = False,
    epsilon: float = FEASIBILITY_EPSILON,
) -> SplitResult:
    """Run the exact fixed-order Split using fused Triton CUDA kernels."""
    if not depot_xy.is_cuda or not node_xy.is_cuda or not giant_tours.is_cuda:
        raise RuntimeError("The Triton Split backend requires CUDA tensors")
    if node_xy.dtype != torch.float32 or depot_xy.dtype != torch.float32:
        raise RuntimeError("The Triton Split backend currently requires FP32 coordinates")
    if giant_tours.dtype != torch.long:
        raise RuntimeError("The Triton Split backend requires int64 giant tours")
    if epsilon != FEASIBILITY_EPSILON:
        raise RuntimeError(
            "The validated Triton backend requires the official MVMoE "
            f"feasibility tolerance epsilon={FEASIBILITY_EPSILON:g}"
        )

    batch, n, _ = node_xy.shape
    if n < 1 or n > 128:
        raise RuntimeError("The validated Triton backend supports 1 <= n <= 128")
    pomo = giant_tours.size(1)
    rows = batch * pomo
    device, dtype = node_xy.device, node_xy.dtype

    xy_idx = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, -1, 2)
    ordered_xy = (
        node_xy[:, None]
        .expand(-1, pomo, -1, -1)
        .gather(2, xy_idx)
        .reshape(rows, n, 2)
        .contiguous()
    )
    demand_idx = giant_tours - 1
    ordered_demand = (
        spec.demand[:, None]
        .expand(-1, pomo, -1)
        .gather(2, demand_idx)
        .reshape(rows, n)
        .contiguous()
    )
    mandatory_breaks = mandatory_breaks.reshape(rows, n)
    positions = torch.arange(n, device=device)[None, :].expand(rows, -1)
    mandatory_positions = torch.where(
        mandatory_breaks, positions, torch.zeros_like(positions)
    )
    last_mandatory_start = torch.cummax(mandatory_positions, dim=1).values.contiguous()
    depot = depot_xy[:, None].expand(-1, pomo, -1, -1).reshape(rows, 1, 2)

    depot_dist_raw = (ordered_xy - depot).norm(p=2, dim=-1).contiguous()
    between_raw = (
        (ordered_xy[:, 1:] - ordered_xy[:, :-1]).norm(p=2, dim=-1).contiguous()
    )
    depot_dist_cost = _rounded_distance(depot_dist_raw, spec.loc_scaler).contiguous()
    between_cost = _rounded_distance(between_raw, spec.loc_scaler).contiguous()

    capacity = _as_batch_vector(spec.capacity, batch, dtype, device)
    capacity = capacity[:, None].expand(-1, pomo).reshape(rows).contiguous()
    if spec.backhaul:
        suffix_has_delivery = torch.flip(
            torch.cumsum(
                torch.flip((ordered_demand > 0).to(torch.int64), dims=(1,)), dim=1
            ),
            dims=(1,),
        ) > 0
        initial_load = torch.where(
            suffix_has_delivery,
            capacity[:, None].expand(-1, n),
            torch.zeros(rows, n, device=device, dtype=dtype),
        ).contiguous()
    else:
        initial_load = capacity[:, None].expand(-1, n).contiguous()

    dummy = torch.empty(1, device=device, dtype=dtype)
    if spec.has_route_limit:
        route_limit = _as_batch_vector(spec.route_limit, batch, dtype, device)
        route_limit = route_limit[:, None].expand(-1, pomo).reshape(rows).contiguous()
    else:
        route_limit = dummy

    if spec.has_time_windows:
        ordered_service = (
            spec.service_time[:, None]
            .expand(-1, pomo, -1)
            .gather(2, demand_idx)
            .reshape(rows, n)
            .contiguous()
        )
        ordered_tw_start = (
            spec.tw_start[:, None]
            .expand(-1, pomo, -1)
            .gather(2, demand_idx)
            .reshape(rows, n)
            .contiguous()
        )
        ordered_tw_end = (
            spec.tw_end[:, None]
            .expand(-1, pomo, -1)
            .gather(2, demand_idx)
            .reshape(rows, n)
            .contiguous()
        )
        depot_start = _as_batch_vector(spec.depot_start, batch, dtype, device)
        depot_end = _as_batch_vector(spec.depot_end, batch, dtype, device)
        speed = _as_batch_vector(spec.speed, batch, dtype, device)
        depot_start = depot_start[:, None].expand(-1, pomo).reshape(rows).contiguous()
        depot_end = depot_end[:, None].expand(-1, pomo).reshape(rows).contiguous()
        speed = speed[:, None].expand(-1, pomo).reshape(rows).contiguous()
    else:
        ordered_service = ordered_tw_start = ordered_tw_end = dummy
        depot_start = depot_end = speed = dummy

    edge_costs = torch.empty((rows, n, n), device=device, dtype=dtype)
    block = triton.next_power_of_2(n + 1)
    _segment_edges_kernel[(rows,)](
        ordered_demand,
        initial_load,
        last_mandatory_start,
        depot_dist_raw,
        between_raw,
        depot_dist_cost,
        between_cost,
        capacity,
        route_limit,
        ordered_service,
        ordered_tw_start,
        ordered_tw_end,
        depot_start,
        depot_end,
        speed,
        edge_costs,
        N=n,
        BLOCK=block,
        HAS_ROUTE_LIMIT=spec.has_route_limit,
        HAS_TIME_WINDOWS=spec.has_time_windows,
        OPEN_ROUTE=spec.open_route,
        EPSILON=epsilon,
        num_warps=4,
    )

    final_costs = torch.empty(rows, device=device, dtype=dtype)
    final_route_counts = torch.empty(rows, device=device, dtype=torch.int32)
    predecessors = torch.empty((rows, n + 1), device=device, dtype=torch.int32)
    _split_dp_kernel[(rows,)](
        edge_costs,
        final_costs,
        final_route_counts,
        predecessors,
        N=n,
        BLOCK=block,
        num_warps=4,
    )

    costs = final_costs.reshape(batch, pomo)
    feasible = torch.isfinite(costs)
    counts = final_route_counts.to(torch.long).reshape(batch, pomo)
    pred_out: Optional[torch.Tensor] = None
    if return_predecessors:
        pred_out = predecessors.to(torch.long).reshape(batch, pomo, n + 1)
    return SplitResult(
        costs=costs,
        feasible=feasible,
        route_counts=counts,
        predecessors=pred_out,
    )
