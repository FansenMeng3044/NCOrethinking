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
from split_envs import MVMoEInstanceAdapter
from tests_split.test_exact_split import brute_force, make_instance
from utils import get_env


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton Split validation requires CUDA"
)


def _official_instance(problem, problem_size, batch=2):
    cls = get_env(problem)[0]
    source = cls(
        problem_size=problem_size,
        pomo_size=min(problem_size, 8),
        device=torch.device("cuda"),
    )
    data = source.get_random_problems(batch, problem_size, normalized=True)
    if isinstance(data, torch.Tensor):
        data = data.cuda()
    else:
        data = tuple(value.cuda() for value in data)
    source.load_problems(batch, problems=data)
    adapted = MVMoEInstanceAdapter.from_official_env(source)
    return adapted.policy_view.depot_xy, adapted.policy_view.node_xy, adapted.split_view


def _representative_tours(spec, problem_size, pomo=8):
    tours = []
    generator = torch.Generator(device="cpu").manual_seed(1729 + problem_size)
    for batch_index in range(spec.demand.size(0)):
        batch_tours = []
        demand = spec.demand[batch_index].cpu()
        for _ in range(pomo):
            if spec.backhaul:
                linehauls = torch.nonzero(demand > 0).flatten() + 1
                backhauls = torch.nonzero(demand < 0).flatten() + 1
                linehauls = linehauls[torch.randperm(len(linehauls), generator=generator)]
                backhauls = backhauls[torch.randperm(len(backhauls), generator=generator)]
                tour = torch.cat((linehauls, backhauls))
            else:
                tour = torch.randperm(problem_size, generator=generator) + 1
            batch_tours.append(tour)
        tours.append(torch.stack(batch_tours))
    return torch.stack(tours).to(device=spec.demand.device, dtype=torch.long)


def _assert_same_result(depot, nodes, tours, spec, mandatory_breaks=None):
    reference = split_giant_tours(
        depot,
        nodes,
        tours,
        spec,
        mandatory_breaks=mandatory_breaks,
        return_predecessors=True,
        backend="reference",
    )
    fused = split_giant_tours(
        depot,
        nodes,
        tours,
        spec,
        mandatory_breaks=mandatory_breaks,
        return_predecessors=True,
        backend="triton",
    )
    torch.cuda.synchronize()
    assert torch.equal(fused.feasible, reference.feasible)
    assert torch.equal(fused.route_counts, reference.route_counts)
    finite = reference.feasible
    assert torch.equal(fused.costs, reference.costs)
    assert torch.equal(fused.predecessors[finite], reference.predecessors[finite])
    for batch_index, pomo_index in finite.nonzero().tolist():
        routes = reconstruct_routes(
            tours[batch_index, pomo_index], fused.predecessors[batch_index, pomo_index]
        )
        replay = verify_routes(depot, nodes, routes, spec, batch_index=batch_index)
        assert replay.feasible, replay.reason
        assert replay.cost == pytest.approx(
            fused.costs[batch_index, pomo_index].item(), abs=2e-5
        )


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
@pytest.mark.parametrize("problem_size", (50, 100))
def test_triton_matches_reference_on_all_official_environments(problem, problem_size):
    torch.manual_seed(314159)
    depot, nodes, spec = _official_instance(problem, problem_size)
    tours = _representative_tours(spec, problem_size)
    _assert_same_result(depot, nodes, tours, spec)


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
def test_triton_matches_reference_with_rounded_objective(problem):
    from dataclasses import replace

    torch.manual_seed(271828)
    depot, nodes, spec = _official_instance(problem, 50, batch=1)
    spec = replace(spec, loc_scaler=100.0)
    tours = _representative_tours(spec, 50, pomo=4)
    _assert_same_result(depot, nodes, tours, spec)


def test_triton_enforces_mandatory_breaks_identically():
    torch.manual_seed(161803)
    depot, nodes, spec = _official_instance("VRPB", 50, batch=1)
    tours = _representative_tours(spec, 50, pomo=4)
    mandatory = torch.zeros_like(tours, dtype=torch.bool)
    mandatory[:, :, 0] = True
    mandatory[:, :, 10] = True
    mandatory[:, :, 25] = True
    _assert_same_result(depot, nodes, tours, spec, mandatory_breaks=mandatory)


def test_triton_matches_official_generator_at_tw_boundary():
    assert FEASIBILITY_EPSILON == 1e-5
    depot = torch.tensor([[[0.0, 0.0]]], device="cuda")
    nodes = torch.tensor([[[0.5, 0.0]]], device="cuda")
    tours = torch.tensor([[[1]]], dtype=torch.long, device="cuda")
    spec = ConstraintSpec(
        problem="VRPTW",
        demand=torch.tensor([[0.1]], device="cuda"),
        capacity=torch.tensor([1.0], device="cuda"),
        has_time_windows=True,
        service_time=torch.tensor([[0.0]], device="cuda"),
        tw_start=torch.tensor([[0.0]], device="cuda"),
        tw_end=torch.tensor([[1.0]], device="cuda"),
        depot_start=torch.tensor([0.0], device="cuda"),
        depot_end=torch.tensor([1.0 - 5e-6], device="cuda"),
        speed=torch.tensor([1.0], device="cuda"),
    )
    _assert_same_result(depot, nodes, tours, spec)
    fused = split_giant_tours(depot, nodes, tours, spec, backend="triton")
    assert fused.feasible.item()


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
@pytest.mark.parametrize("loc_scaler", (None, 100.0))
def test_triton_matches_exhaustive_partition_oracle(problem, loc_scaler):
    from dataclasses import replace

    depot_cpu, nodes_cpu, spec_cpu = make_instance(problem, loc_scaler)
    depot_cpu = depot_cpu.float()
    nodes_cpu = nodes_cpu.float()
    converted = {}
    for name in (
        "demand", "capacity", "route_limit", "service_time", "tw_start",
        "tw_end", "depot_start", "depot_end", "speed",
    ):
        value = getattr(spec_cpu, name)
        converted[name] = value.float() if isinstance(value, torch.Tensor) else value
    spec_cpu = replace(spec_cpu, **converted)
    tours_cpu = torch.tensor([[
        [1, 2, 3, 4, 5, 6],
        [2, 1, 3, 5, 4, 6],
        [3, 1, 2, 6, 5, 4],
    ]], dtype=torch.long)
    fused = split_giant_tours(
        depot_cpu.cuda(), nodes_cpu.cuda(), tours_cpu.cuda(), spec_cpu.to(torch.device("cuda")),
        return_predecessors=True, backend="triton",
    )
    torch.cuda.synchronize()
    for pomo_index in range(tours_cpu.size(1)):
        tour = tours_cpu[0, pomo_index].tolist()
        brute_cost, brute_routes, brute_result = brute_force(
            depot_cpu, nodes_cpu, tour, spec_cpu
        )
        if math.isfinite(brute_cost):
            assert fused.feasible[0, pomo_index]
            assert fused.costs[0, pomo_index].item() == pytest.approx(
                brute_cost, abs=2e-5
            )
            assert fused.route_counts[0, pomo_index].item() == brute_result.route_count
            routes = reconstruct_routes(
                tours_cpu[0, pomo_index], fused.predecessors[0, pomo_index].cpu()
            )
            replay = verify_routes(depot_cpu, nodes_cpu, routes, spec_cpu)
            assert replay.feasible, replay.reason
            assert replay.cost == pytest.approx(brute_cost, abs=2e-5)
        else:
            assert brute_routes is None
            assert not fused.feasible[0, pomo_index]
            assert math.isinf(fused.costs[0, pomo_index].item())
            assert fused.route_counts[0, pomo_index].item() == -1
