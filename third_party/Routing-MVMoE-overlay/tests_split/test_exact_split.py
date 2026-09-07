import itertools
import math

import pytest
import torch

from split import (
    ALL_PROBLEMS,
    FEASIBILITY_EPSILON,
    ConstraintSpec,
    reconstruct_routes,
    split_giant_tours,
    verify_routes,
)
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


def brute_force(depot_xy, node_xy, tour, spec, mandatory_breaks=None):
    n = len(tour)
    mandatory_breaks = mandatory_breaks or [False] * n
    best = (float("inf"), None, None)
    for bits in itertools.product((0, 1), repeat=n - 1):
        if any(mandatory_breaks[position] and not bits[position - 1] for position in range(1, n)):
            continue
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
        [2, 1, 3, 5, 4, 6],
        [3, 1, 2, 6, 5, 4],
    ]], dtype=torch.long)
    result = split_giant_tours(
        depot_xy, node_xy, tours, spec, return_predecessors=True,
    )
    for p in range(tours.size(1)):
        tour = tours[0, p].tolist()
        brute_cost, _, brute_result = brute_force(
            depot_xy, node_xy, tour, spec
        )
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


def test_mandatory_b_breaks_are_enforced_by_full_split():
    depot_xy, node_xy, spec = make_instance("VRPB")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    mandatory = torch.tensor([[[True, False, True, False, True, False]]])
    result = split_giant_tours(
        depot_xy, node_xy, tour, spec,
        mandatory_breaks=mandatory, return_predecessors=True,
    )
    routes = reconstruct_routes(tour[0, 0], result.predecessors[0, 0])
    starts = []
    position = 0
    for route in routes:
        starts.append(position)
        position += len(route)
    assert {0, 2, 4}.issubset(starts)


def test_rejects_missing_initial_mandatory_break():
    depot_xy, node_xy, spec = make_instance("CVRP")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    with pytest.raises(ValueError, match="first customer"):
        split_giant_tours(
            depot_xy, node_xy, tour, spec,
            mandatory_breaks=torch.zeros_like(tour, dtype=torch.bool),
        )


def test_backhaul_is_rechecked_by_split_without_decoder_boundaries():
    depot_xy, node_xy, spec = make_instance("VRPB")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    result = split_giant_tours(
        depot_xy, node_xy, tour, spec, return_predecessors=True
    )
    assert result.feasible.item()
    routes = reconstruct_routes(tour[0, 0], result.predecessors[0, 0])
    replay = verify_routes(depot_xy, node_xy, routes, spec)
    assert replay.feasible, replay.reason


def test_split_rejects_backhaul_before_remaining_linehauls():
    depot_xy, node_xy, spec = make_instance("VRPB")
    # Customer 4 is a backhaul. A route cannot start there while positive
    # linehaul demand remains globally unserved under MVMoE's full-load reset.
    tour = torch.tensor([[[4, 1, 2, 3, 5, 6]]])
    result = split_giant_tours(depot_xy, node_xy, tour, spec)
    assert not result.feasible.item()
    assert math.isinf(result.costs.item())


def test_length_problem_needs_no_decoder_boundaries():
    depot_xy, node_xy, spec = make_instance("VRPL")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    result = split_giant_tours(
        depot_xy, node_xy, tour, spec, return_predecessors=True
    )
    assert result.feasible.item()
    routes = reconstruct_routes(tour[0, 0], result.predecessors[0, 0])
    replay = verify_routes(depot_xy, node_xy, routes, spec)
    assert replay.feasible, replay.reason


def test_open_route_omits_return_edge_and_return_constraints():
    depot_xy, node_xy, closed = make_instance("VRPTW")
    _, _, open_spec = make_instance("OVRPTW")
    tour = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    closed_result = split_giant_tours(depot_xy, node_xy, tour, closed)
    open_result = split_giant_tours(depot_xy, node_xy, tour, open_spec)
    assert open_result.feasible.item()
    assert open_result.costs.item() <= closed_result.costs.item()


def test_default_tolerance_matches_official_generator_at_tw_boundary():
    """A singleton accepted by the official generator must be accepted by Split."""
    assert FEASIBILITY_EPSILON == 1e-5
    depot_xy = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
    node_xy = torch.tensor([[[0.5, 0.0]]], dtype=torch.float64)
    tour = torch.tensor([[[1]]], dtype=torch.long)
    # Round trip is 1.0, exceeding the depot horizon by 5e-6: accepted by
    # MVMoE's official tolerance, but rejected by the old 1e-6 boundary.
    spec = ConstraintSpec(
        problem="VRPTW",
        demand=torch.tensor([[0.1]], dtype=torch.float64),
        capacity=torch.tensor([1.0], dtype=torch.float64),
        has_time_windows=True,
        service_time=torch.tensor([[0.0]], dtype=torch.float64),
        tw_start=torch.tensor([[0.0]], dtype=torch.float64),
        tw_end=torch.tensor([[1.0]], dtype=torch.float64),
        depot_start=torch.tensor([0.0], dtype=torch.float64),
        depot_end=torch.tensor([1.0 - 5e-6], dtype=torch.float64),
        speed=torch.tensor([1.0], dtype=torch.float64),
    )

    official = split_giant_tours(
        depot_xy, node_xy, tour, spec, return_predecessors=True
    )
    strict = split_giant_tours(
        depot_xy, node_xy, tour, spec, epsilon=1e-6
    )
    assert official.feasible.item()
    assert not strict.feasible.item()
    routes = reconstruct_routes(tour[0, 0], official.predecessors[0, 0])
    replay = verify_routes(depot_xy, node_xy, routes, spec)
    assert replay.feasible, replay.reason
