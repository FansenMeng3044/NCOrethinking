from dataclasses import fields

import pytest
import torch

from split.constraints import flags_from_problem
from split_envs import GiantTourEnv, MVMoEInstanceAdapter
from split_models import MVMoE4ELSplit, MVMoE4ESplit, POMOMTLSplit
from utils import get_env


TRAIN_PROBLEMS = ("CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW")
MODEL_CLASSES = (POMOMTLSplit, MVMoE4ESplit, MVMoE4ELSplit)


def params(device=torch.device("cpu")):
    return {
        "embedding_dim": 16,
        "sqrt_embedding_dim": 4.0,
        "encoder_layer_num": 1,
        "decoder_layer_num": 1,
        "qkv_dim": 4,
        "head_num": 2,
        "logit_clipping": 10.0,
        "ff_hidden_dim": 32,
        "num_experts": 4,
        "eval_type": "argmax",
        "norm": "layer",
        "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Dec"],
        "problem": "Train_ALL",
        "topk": 2,
        "routing_level": "node",
        "routing_method": "input_choice",
        "device": device,
    }


def official_env(problem, batch=2, n=20, pomo=8):
    cls = get_env(problem)[0]
    env = cls(problem_size=n, pomo_size=pomo, device=torch.device("cpu"))
    data = env.get_random_problems(batch, n, normalized=True)
    env.load_problems(batch, problems=data)
    return env


def test_infeasible_candidates_never_enter_the_pomo_baseline():
    from SplitTrainer import valid_pomo_reinforce_loss

    reward = torch.tensor([[-3.0, float("-inf"), -5.0]])
    log_prob = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    loss, valid = valid_pomo_reinforce_loss(reward, log_prob)
    assert valid.tolist() == [[True, False, True]]
    assert torch.isfinite(loss)
    loss.backward()
    assert log_prob.grad[0, 1].item() == 0.0
    assert torch.isfinite(log_prob.grad).all()


def test_no_feasible_candidate_fails_loudly_without_relaxation():
    from SplitTrainer import NoFeasibleCandidateError, valid_pomo_reinforce_loss

    with pytest.raises(NoFeasibleCandidateError):
        valid_pomo_reinforce_loss(
            torch.tensor([[float("-inf"), float("-inf")]]),
            torch.tensor([[-1.0, -2.0]]),
        )


def test_untrainable_rows_are_skipped_without_relaxing_constraints():
    from SplitTrainer import valid_pomo_reinforce_loss

    reward = torch.tensor([
        [float("-inf"), float("-inf")],
        [-3.0, -5.0],
    ])
    log_prob = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True
    )
    loss, valid = valid_pomo_reinforce_loss(reward, log_prob)
    assert valid.tolist() == [[False, False], [True, True]]
    assert torch.isfinite(loss)
    loss.backward()
    assert log_prob.grad[0].eq(0).all()
    assert torch.isfinite(log_prob.grad).all()


@pytest.mark.parametrize("problem", TRAIN_PROBLEMS)
def test_adapter_separates_policy_and_constraint_views(problem):
    source = official_env(problem)
    adapted = MVMoEInstanceAdapter.from_official_env(source)
    assert {field.name for field in fields(adapted.policy_view)} == {"depot_xy", "node_xy"}
    assert adapted.policy_view.node_xy.shape[-1] == 2
    assert (
        adapted.split_view.open_route,
        adapted.split_view.backhaul,
        adapted.split_view.has_route_limit,
        adapted.split_view.has_time_windows,
    ) == flags_from_problem(problem)


def test_action_mask_is_only_depot_plus_visited_customers():
    env = GiantTourEnv.from_official_env(official_env("VRPBLTW"), pomo_size=8)
    _, _, _ = env.reset()
    assert torch.isneginf(env.ninf_mask[:, :, 0]).all()
    assert (env.ninf_mask[:, :, 1:] == 0).all()
    first = env.START_NODE
    env.step(first)
    for b in range(env.batch_size):
        for p in range(env.pomo_size):
            masked = torch.nonzero(torch.isneginf(env.ninf_mask[b, p])).flatten().tolist()
            assert masked == [0, int(first[b, p])]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_policy_is_identical_when_constraints_change_but_xy_does_not(model_class):
    torch.manual_seed(7)
    cvrp = official_env("CVRP", batch=1)
    vrptw = official_env("VRPTW", batch=1)
    # Force exactly the same policy observation while retaining different constraints.
    vrptw.depot_node_xy = cvrp.depot_node_xy.clone()
    model = model_class(**params()).eval()

    tours = []
    for source in (cvrp, vrptw):
        torch.manual_seed(99)
        env = GiantTourEnv.from_official_env(source, pomo_size=8)
        reset, _, _ = env.reset()
        model.pre_forward(reset)
        state, _, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, _, done = env.step(selected)
        tours.append(env.selected_node_list.clone())
    assert torch.equal(tours[0], tours[1])


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("problem", TRAIN_PROBLEMS)
def test_three_models_have_finite_forward_backward_on_all_six_training_tasks(model_class, problem):
    from SplitTrainer import valid_pomo_reinforce_loss

    torch.manual_seed(123)
    source = official_env(problem, batch=2, n=20, pomo=8)
    env = GiantTourEnv.from_official_env(source, pomo_size=8)
    model = model_class(**params()).train()
    reset, _, _ = env.reset()
    model.pre_forward(reset)
    state, reward, done = env.pre_step()
    probs = []
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        probs.append(prob)
    log_prob = torch.stack(probs, dim=2).log().sum(dim=2)
    loss, valid = valid_pomo_reinforce_loss(reward, log_prob)
    aux = model.aux_loss
    if isinstance(aux, torch.Tensor):
        loss = loss + aux.mean()
    assert valid.any(dim=1).all()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.encoder.embedding_node.input_size == 2 if hasattr(model.encoder.embedding_node, "input_size") else model.encoder.embedding_node.in_features == 2
    assert model.decoder.Wq_first.in_features == params()["embedding_dim"]
    assert model.decoder.Wq_last.in_features == params()["embedding_dim"]
