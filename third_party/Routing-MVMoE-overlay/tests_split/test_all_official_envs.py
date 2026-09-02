import pytest
import torch

from split import ALL_PROBLEMS, flags_from_problem
from split_envs import GiantTourEnv, MVMoEInstanceAdapter
from utils import get_env


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
def test_every_official_environment_adapts_without_modification(problem):
    cls = get_env(problem)[0]
    source = cls(problem_size=20, pomo_size=8, device=torch.device("cpu"))
    data = source.get_random_problems(2, 20, normalized=True)
    source.load_problems(2, problems=data)
    adapted = MVMoEInstanceAdapter.from_official_env(source)
    assert adapted.policy_view.depot_xy.shape == (2, 1, 2)
    assert adapted.policy_view.node_xy.shape == (2, 20, 2)
    assert (
        adapted.split_view.open_route,
        adapted.split_view.backhaul,
        adapted.split_view.has_route_limit,
        adapted.split_view.has_time_windows,
    ) == flags_from_problem(problem)
    env = GiantTourEnv(adapted, pomo_size=8)
    reset, _, _ = env.reset()
    assert tuple(vars(reset)) == ("depot_xy", "node_xy")
