import itertools
import math

import pytest
import torch

from split import ALL_PROBLEMS, ConstraintSpec, reconstruct_routes, split_giant_tours, verify_routes
from split.constraints import flags_from_problem


def make_instance(problem, loc_scaler=None):
    open_route, backhaul, has_limit, has_tw = flags_from_problem(problem)
    node_xy = torch.tensor([[
        [0.10, 0.20], [0.35, 0.10], [0.70, 0.18],
        [0.82, 0.65], [0.28, 0.82], [0.55, 0.52],
    ]], dtype=torch.float64)
    depot_xy = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    demand = torch.tensor(
        [[0.35, 0.40, 0.20, -0.25, -0.30, -0.15]]
        if backhaul else [[0.35, 0.40, 0.20, 0.25, 0.30, 0.15]],
        dtype=torch.float64,
    )
    kwargs = {}
    if has_limit:
        kwargs["route_limit"] = torch.tensor([2.1], dtype=torch.float64)
    if has_tw:
        kwargs.update(
            service_time=torch.tensor([[0.03] * 6], dtype=torch.float64),
            tw_start=torch.tensor([[0.0, 0.1, 0.0, 0.2, 0.0, 0.1]], dtype=torch.float64),
            tw_end=torch.tensor([[1.3, 1.5, 1.7, 2.0, 2.1, 2.3]], dtype=torch.float64),
            depot_start=torch.tensor([0.0], dtype=torch.float64),
            depot_end=torch.tensor([3.0], dtype=torch.float64),
            speed=torch.tensor([1.0], dtype=torch.float64),
        )
    spec = ConstraintSpec(
        problem=problem,
        demand=demand,
        capacity=torch.tensor([1.0], dtype=torch.float64),
        open_route=open_route,
        backhaul=backhaul,
        has_route_limit=has_limit,
        has_time_windows=has_tw,
        loc_scaler=loc_scaler,
        **kwargs,
    )
    return depot_xy, node_xy, spec


def brute_force(depot_xy, node_xy, tour, spec):
    n = len(tour)
    best = (float("inf"), None, None)
    for bits in itertools.product((0, 1), repeat=n - 1):
        routes, start = [], 0
        for position, split_here in enumerate(bits, start=1):
            if split_here:
                routes.append(tour[start:position])
                start = position
        routes.append(tour[start:])
        result = verify_routes(depot_xy, node_xy, routes, spec)
        if result.feasible and (result.cost, result.route_count) < (best[0], len(best[1]) if best[1] else n + 1):
            best = (result.cost, routes, result)
    return best


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
@pytest.mark.parametrize("loc_scaler", [None, 100.0])
def test_dynamic_program_matches_exhaustive_partitions(problem, loc_scaler):
    depot_xy, node_xy, spec = make_instance(problem, loc_scaler)
    tours = torch.tensor([[
        [1, 2, 3, 4, 5, 6],
        [2, 5, 1, 6, 3, 4],
        [4, 1, 2, 5, 3, 6],
    ]], dtype=torch.long)
    result = split_giant_tours(depot_xy, node_xy, tours, spec, return_predecessors=True)
    for p in range(tours.size(1)):
        tour = tours[0, p].tolist()
        brute_cost, _, brute_result = brute_force(depot_xy, node_xy, tour, spec)
        if math.isfinite(brute_cost):
            assert result.feasible[0, p]
            assert result.costs[0, p].item() == pytest.approx(brute_cost, abs=1e-9)
            routes = reconstruct_routes(tours[0, p], result.predecessors[0, p])
            replay = verify_routes(depot_xy, node_xy, routes, spec)
            assert replay.feasible, replay.reason
            assert replay.cost == pytest.approx(result.costs[0, p].item(), abs=1e-9)
            assert replay.route_count == result.route_counts[0, p].item()
            assert result.costs[0, p].item() <= brute_result.cost + 1e-9
        else:
            assert not result.feasible[0, p]
            assert math.isinf(result.costs[0, p].item())
            assert result.route_counts[0, p].item() == -1


def test_backhaul_start_semantics_match_official_environment():
    depot_xy, node_xy, spec = make_instance("VRPB")
    # Customer 4 is a pickup. With unserved deliveries in the suffix, the
    # official environment starts full, so 4 cannot be the first route action.
    tour = torch.tensor([[[4, 1, 2, 3, 5, 6]]])
    result = split_giant_tours(depot_xy, node_xy, tour, spec)
    assert not result.feasible.item()


def test_open_route_omits_return_edge_and_return_constraints():
    depot_xy, node_xy, closed = make_instance("VRPTW")
    _, _, open_spec = make_instance("OVRPTW")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    closed_result = split_giant_tours(depot_xy, node_xy, tour, closed)
    open_result = split_giant_tours(depot_xy, node_xy, tour, open_spec)
    assert open_result.feasible.item()
    assert open_result.costs.item() <= closed_result.costs.item()
