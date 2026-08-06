from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SplitResult:
    """Costs and optional predecessor labels for a batch of giant tours."""

    costs: torch.Tensor
    predecessors: Optional[torch.Tensor] = None


def split_giant_tours(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    giant_tours: torch.Tensor,
    capacity: float = 1.0,
    return_predecessors: bool = False,
    epsilon: float = 1e-6,
) -> SplitResult:
    """Optimally split fixed customer permutations under a hard capacity.

    Customer ids in ``giant_tours`` are 1..n because index 0 is reserved for
    the depot inside the Attention Model decoder. The customer order is never
    changed. Complexity is O(batch * n^2).
    """

    if depot_xy.dim() == 3 and depot_xy.size(1) == 1:
        depot_xy = depot_xy[:, 0]
    _validate_inputs(depot_xy, node_xy, node_demand, giant_tours, capacity)

    batch_size, problem_size, _ = node_xy.shape
    gather_index = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, 2)
    ordered_xy = node_xy.gather(1, gather_index)
    ordered_demand = node_demand.gather(1, giant_tours - 1)

    depot_distance = (ordered_xy - depot_xy[:, None, :]).norm(p=2, dim=-1)
    if problem_size > 1:
        consecutive_distance = (ordered_xy[:, 1:] - ordered_xy[:, :-1]).norm(p=2, dim=-1)
        edge_prefix = torch.cat(
            (
                torch.zeros(batch_size, 1, device=node_xy.device, dtype=node_xy.dtype),
                consecutive_distance.cumsum(dim=1),
            ),
            dim=1,
        )
    else:
        edge_prefix = torch.zeros(batch_size, 1, device=node_xy.device, dtype=node_xy.dtype)

    demand_prefix = torch.cat(
        (
            torch.zeros(batch_size, 1, device=node_demand.device, dtype=node_demand.dtype),
            ordered_demand.cumsum(dim=1),
        ),
        dim=1,
    )

    inf = torch.tensor(float("inf"), device=node_xy.device, dtype=node_xy.dtype)
    potential = torch.full(
        (batch_size, problem_size + 1), inf, device=node_xy.device, dtype=node_xy.dtype
    )
    potential[:, 0] = 0
    predecessors = None
    if return_predecessors:
        predecessors = torch.full(
            (batch_size, problem_size + 1),
            -1,
            device=node_xy.device,
            dtype=torch.long,
        )

    # Auxiliary edge (start, end) is depot -> pi[start:end] -> depot.
    for end in range(1, problem_size + 1):
        segment_load = demand_prefix[:, end, None] - demand_prefix[:, :end]
        internal_distance = edge_prefix[:, end - 1, None] - edge_prefix[:, :end]
        segment_cost = (
            depot_distance[:, :end]
            + internal_distance
            + depot_distance[:, end - 1, None]
        )
        candidate = potential[:, :end] + segment_cost
        candidate = candidate.masked_fill(segment_load > capacity + epsilon, inf)
        best_cost, best_start = candidate.min(dim=1)
        potential[:, end] = best_cost
        if predecessors is not None:
            predecessors[:, end] = best_start

    final_cost = potential[:, problem_size]
    if not torch.isfinite(final_cost).all():
        raise ValueError(
            "Split found no feasible solution; check customer demands and capacity"
        )
    return SplitResult(final_cost, predecessors)


def raw_giant_tour_cost(
    depot_xy: torch.Tensor, node_xy: torch.Tensor, giant_tours: torch.Tensor
) -> torch.Tensor:
    """Distance of depot -> giant tour -> depot, ignoring capacity."""

    if depot_xy.dim() == 3 and depot_xy.size(1) == 1:
        depot_xy = depot_xy[:, 0]
    gather_index = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, 2)
    ordered_xy = node_xy.gather(1, gather_index)
    internal = (
        (ordered_xy[:, 1:] - ordered_xy[:, :-1]).norm(p=2, dim=-1).sum(dim=1)
        if ordered_xy.size(1) > 1
        else ordered_xy.new_zeros(ordered_xy.size(0))
    )
    return (
        internal
        + (ordered_xy[:, 0] - depot_xy).norm(p=2, dim=-1)
        + (ordered_xy[:, -1] - depot_xy).norm(p=2, dim=-1)
    )


def reconstruct_routes(
    giant_tour: torch.Tensor, predecessors: torch.Tensor
) -> List[List[int]]:
    """Recover capacity-feasible customer routes from Split predecessors."""

    if giant_tour.dim() != 1 or predecessors.dim() != 1:
        raise ValueError("giant_tour and predecessors must be rank-1 tensors")
    if predecessors.numel() != giant_tour.numel() + 1:
        raise ValueError("predecessors must have n+1 entries")

    tour = giant_tour.detach().cpu().tolist()
    pred = predecessors.detach().cpu().tolist()
    routes = []
    end = len(tour)
    while end > 0:
        begin = pred[end]
        if begin < 0 or begin >= end:
            raise ValueError("invalid Split predecessor chain")
        routes.append(tour[begin:end])
        end = begin
    routes.reverse()
    return routes


def _validate_inputs(depot_xy, node_xy, node_demand, giant_tours, capacity):
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if depot_xy.dim() != 2 or depot_xy.size(1) != 2:
        raise ValueError("depot_xy must have shape (batch, 2) or (batch, 1, 2)")
    if node_xy.dim() != 3 or node_xy.size(2) != 2:
        raise ValueError("node_xy must have shape (batch, n, 2)")
    if node_demand.shape != node_xy.shape[:2]:
        raise ValueError("node_demand must have shape (batch, n)")
    if giant_tours.shape != node_demand.shape:
        raise ValueError("giant_tours must have shape (batch, n)")
    if depot_xy.size(0) != node_xy.size(0):
        raise ValueError("batch dimensions do not match")

    n = node_xy.size(1)
    expected = torch.arange(1, n + 1, device=giant_tours.device, dtype=giant_tours.dtype)
    if not torch.equal(giant_tours.sort(dim=1).values, expected[None].expand_as(giant_tours)):
        raise ValueError("each giant tour must be a permutation of customer ids 1..n")
    if (node_demand < 0).any().item():
        raise ValueError("customer demands must be non-negative")
    if (node_demand > capacity + 1e-6).any().item():
        raise ValueError("a customer demand exceeds vehicle capacity")
