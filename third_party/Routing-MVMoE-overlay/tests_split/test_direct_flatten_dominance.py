import pytest
import torch

from models.MTLModel import MTLModel
from split import ALL_PROBLEMS, reconstruct_routes, split_giant_tours, verify_routes
from split_envs import MVMoEInstanceAdapter
from utils import get_env


def direct_params():
    return {
        "embedding_dim": 16,
        "sqrt_embedding_dim": 4.0,
        "encoder_layer_num": 1,
        "decoder_layer_num": 1,
        "qkv_dim": 4,
        "head_num": 2,
        "logit_clipping": 10.0,
        "ff_hidden_dim": 32,
        "num_experts": 1,
        "eval_type": "argmax",
        "norm": "layer",
        "norm_loc": "norm_last",
        "expert_loc": [],
        "problem": "ALL",
        "topk": 1,
        "routing_level": "node",
        "routing_method": "input_choice",
        "device": torch.device("cpu"),
    }


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
def test_split_is_no_worse_than_boundaries_of_official_feasible_direct_routes(problem):
    """Cross-check Split semantics against the untouched official transitions."""
    torch.manual_seed(31415)
    source_cls = get_env(problem)[0]
    source = source_cls(problem_size=20, pomo_size=8, device=torch.device("cpu"))
    data = source.get_random_problems(1, 20, normalized=True)
    source.load_problems(1, problems=data)
    reset, _, _ = source.reset()
    model = MTLModel(**direct_params()).eval()
    model.pre_forward(reset)
    state, reward, done = source.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done = source.step(selected)

    adapted = MVMoEInstanceAdapter.from_official_env(source)
    direct_sequences = source.selected_node_list
    tours = []
    mandatory_breaks = []
    for p in range(source.pomo_size):
        sequence = direct_sequences[0, p].tolist()
        flattened = direct_sequences[0, p][direct_sequences[0, p] != 0]
        assert flattened.numel() == 20
        assert torch.equal(flattened.sort().values, torch.arange(1, 21))
        tours.append(flattened)
        starts = []
        previous = 0
        for node in sequence:
            if node != 0:
                starts.append(previous == 0)
            previous = node
        assert len(starts) == 20 and starts[0]
        mandatory_breaks.append(torch.tensor(starts, dtype=torch.bool))
    tours = torch.stack(tours, dim=0).unsqueeze(0)
    mandatory_breaks = torch.stack(mandatory_breaks, dim=0).unsqueeze(0)
    split = split_giant_tours(
        adapted.policy_view.depot_xy,
        adapted.policy_view.node_xy,
        tours,
        adapted.split_view,
        mandatory_breaks=mandatory_breaks,
        return_predecessors=True,
    )
    assert split.feasible.all()
    direct_cost = -reward
    assert torch.all(split.costs <= direct_cost + 2e-5)
    for p in range(source.pomo_size):
        routes = reconstruct_routes(tours[0, p], split.predecessors[0, p])
        replay = verify_routes(
            adapted.policy_view.depot_xy,
            adapted.policy_view.node_xy,
            routes,
            adapted.split_view,
        )
        assert replay.feasible, replay.reason
        assert replay.cost == pytest.approx(split.costs[0, p].item(), abs=2e-5)
