import pytest
import torch

from split import ALL_PROBLEMS, flags_from_problem
from split_envs import GiantTourEnv, MVMoEInstanceAdapter
from utils import get_env


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
@pytest.mark.parametrize("problem_size", (50, 100))
def test_every_official_environment_adapts_at_both_sizes(problem, problem_size):
    cls = get_env(problem)[0]
    source = cls(problem_size=problem_size, pomo_size=8, device=torch.device("cpu"))
    data = source.get_random_problems(2, problem_size, normalized=True)
    source.load_problems(2, problems=data)
    adapted = MVMoEInstanceAdapter.from_official_env(source)
    assert adapted.policy_view.depot_xy.shape == (2, 1, 2)
    assert adapted.policy_view.node_xy.shape == (2, problem_size, 2)
    assert adapted.policy_view.node_demand.shape == (2, problem_size)
    assert adapted.policy_view.node_tw_start.shape == (2, problem_size)
    assert adapted.policy_view.node_tw_end.shape == (2, problem_size)
    assert torch.equal(
        adapted.policy_view.node_demand, source.depot_node_demand[:, 1:]
    )
    if flags_from_problem(problem)[3]:
        assert torch.equal(
            adapted.policy_view.node_tw_start, source.depot_node_tw_start[:, 1:]
        )
        assert torch.equal(
            adapted.policy_view.node_tw_end, source.depot_node_tw_end[:, 1:]
        )
    else:
        assert torch.count_nonzero(adapted.policy_view.node_tw_start) == 0
        assert torch.count_nonzero(adapted.policy_view.node_tw_end) == 0
    assert (
        adapted.split_view.open_route,
        adapted.split_view.backhaul,
        adapted.split_view.has_route_limit,
        adapted.split_view.has_time_windows,
    ) == flags_from_problem(problem)
    env = GiantTourEnv(adapted, pomo_size=8)
    reset, _, _ = env.reset()
    assert tuple(vars(reset)) == (
        "depot_xy", "node_xy", "node_demand", "node_tw_start", "node_tw_end"
    )
    assert env.step_state.load.shape == (2, 8)
    assert env.step_state.current_time.shape == (2, 8)
    assert env.step_state.length.shape == (2, 8)
    assert env.step_state.open.shape == (2, 8)
    if adapted.split_view.backhaul:
        demand = adapted.split_view.demand[:, None, :].expand(-1, env.pomo_size, -1)
        assert torch.isneginf(env.ninf_mask[:, :, 1:])[demand < 0].all()
        assert not torch.isneginf(env.ninf_mask[:, :, 1:])[demand > 0].any()
    else:
        assert not torch.isneginf(env.ninf_mask[:, :, 1:]).any()
    expected_open = float(flags_from_problem(problem)[0])
    assert torch.equal(env.step_state.open, torch.full_like(env.step_state.open, expected_open))


@pytest.mark.parametrize("problem_size", (50, 100))
def test_backhaul_pomo_starts_match_official_linehaul_pool(problem_size):
    cls = get_env("VRPB")[0]
    source = cls(
        problem_size=problem_size, pomo_size=problem_size, device=torch.device("cpu")
    )
    data = source.get_random_problems(2, problem_size, normalized=True)
    source.load_problems(2, problems=data)
    env = GiantTourEnv.from_official_env(source, pomo_size=problem_size)
    expected = int(problem_size * (1 - source.backhaul_ratio))
    assert env.pomo_size == expected
    starts = env.instance.split_view.demand.gather(1, env.START_NODE - 1)
    assert (starts > 0).all()
