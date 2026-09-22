from dataclasses import fields

import pytest
import torch

from split import ConstraintSpec, verify_routes
from split.constraints import flags_from_problem
from split_envs import AdaptedInstance, GiantTourEnv, MVMoEInstanceAdapter, PolicyView
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


@pytest.mark.parametrize("problem", TRAIN_PROBLEMS)
def test_adapter_separates_policy_and_constraint_views(problem):
    source = official_env(problem)
    adapted = MVMoEInstanceAdapter.from_official_env(source)
    assert {field.name for field in fields(adapted.policy_view)} == {
        "depot_xy", "node_xy", "node_demand", "node_tw_start", "node_tw_end"
    }
    assert adapted.policy_view.node_xy.shape[-1] == 2
    assert adapted.policy_view.node_demand.shape == adapted.policy_view.node_xy.shape[:2]
    assert adapted.policy_view.node_tw_start.shape == adapted.policy_view.node_xy.shape[:2]
    assert adapted.policy_view.node_tw_end.shape == adapted.policy_view.node_xy.shape[:2]
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


def test_backhaul_ordering_mask_preserves_a_feasible_partition_witness():
    source = official_env("VRPB", batch=1, n=20, pomo=1)
    env = GiantTourEnv.from_official_env(source, pomo_size=1)
    _, _, _ = env.reset()
    assert torch.isneginf(env.ninf_mask[:, :, 0]).all()
    demand = env.instance.split_view.demand[0]
    assert (demand[env.START_NODE[0] - 1] > 0).all()
    initial_mask = torch.isneginf(env.ninf_mask[0, 0, 1:])
    assert initial_mask[demand < 0].all()
    assert not initial_mask[demand > 0].any()
    assert env.step_state.load.tolist() == [[1.0]]
    assert env.step_state.current_time.tolist() == [[0.0]]
    assert env.step_state.length.tolist() == [[0.0]]
    assert env.step_state.open.tolist() == [[0.0]]

    selected = env.START_NODE
    selected_demand = demand[selected.item() - 1]
    env.step(selected)
    assert torch.allclose(env.step_state.load, 1.0 - selected_demand.reshape(1, 1))
    assert not torch.isneginf(env.ninf_mask[0, 0, 1:]).all()


def test_length_is_decoder_context_but_not_an_action_mask():
    depot = torch.tensor([[[0.0, 0.0]]])
    nodes = torch.tensor([[[0.4, 0.0], [-0.4, 0.0], [0.0, 0.4]]])
    spec = ConstraintSpec(
        problem="VRPL",
        demand=torch.tensor([[0.2, 0.2, 0.2]]),
        capacity=torch.tensor([1.0]),
        has_route_limit=True,
        route_limit=torch.tensor([1.1]),
    )
    demand = spec.demand
    zeros = torch.zeros_like(demand)
    env = GiantTourEnv(
        AdaptedInstance(PolicyView(depot, nodes, demand, zeros, zeros), spec),
        pomo_size=1,
    )
    env.reset()
    assert env.step_state.length.tolist() == [[0.0]]
    assert not torch.isneginf(env.ninf_mask[0, 0, 1:]).any()
    for customer in (1, 2, 3):
        _, reward, done = env.step(torch.tensor([[customer]]))
    assert done
    assert env.step_state.length.item() > spec.route_limit.item()
    routes = env.get_routes(0, 0)
    assert routes == [[1], [2], [3]]
    replay = verify_routes(depot, nodes, routes, spec)
    assert replay.feasible, replay.reason
    assert torch.isfinite(reward).all()


def test_direct_dynamic_attributes_update_without_constraint_masks():
    source = official_env("OVRPTW", batch=1, n=20, pomo=1)
    env = GiantTourEnv.from_official_env(source, pomo_size=1)
    env.reset()
    selected = env.START_NODE
    customer = selected.item() - 1
    spec = env.instance.split_view
    coord = env.node_xy[0, customer]
    travel = torch.linalg.vector_norm(coord - env.depot_xy[0, 0])
    expected_time = torch.maximum(
        travel / spec.speed[0], spec.tw_start[0, customer]
    ) + spec.service_time[0, customer]
    _, _, done = env.step(selected)
    assert not done
    assert torch.allclose(env.step_state.load, 1.0 - spec.demand[0, customer])
    assert torch.allclose(env.step_state.current_time, expected_time.reshape(1, 1))
    assert torch.allclose(env.step_state.length, travel.reshape(1, 1))
    assert env.step_state.open.tolist() == [[1.0]]
    assert not torch.isneginf(env.ninf_mask[0, 0, 1:]).all()


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("problem", ("CVRP", "VRPTW"))
def test_encoder_receives_original_five_customer_features(model_class, problem):
    torch.manual_seed(7)
    source = official_env(problem, batch=1)
    env = GiantTourEnv.from_official_env(source, pomo_size=8)
    reset, _, _ = env.reset()
    model = model_class(**params()).eval()
    captured = []
    handle = model.encoder.embedding_node.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        model.pre_forward(reset)
    finally:
        handle.remove()
    expected = torch.cat(
        (
            reset.node_xy,
            reset.node_demand[:, :, None],
            reset.node_tw_start[:, :, None],
            reset.node_tw_end[:, :, None],
        ),
        dim=2,
    )
    assert len(captured) == 1
    assert captured[0].shape[-1] == 5
    assert torch.equal(captured[0], expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_decoder_receives_original_four_dynamic_attributes(model_class):
    torch.manual_seed(11)
    source = official_env("OVRPTW", batch=1, pomo=2)
    env = GiantTourEnv.from_official_env(source, pomo_size=2)
    model = model_class(**params()).eval()
    reset, _, _ = env.reset()
    model.pre_forward(reset)
    state, _, _ = env.step(env.START_NODE)
    captured = []
    handle = model.decoder.Wq_last.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        model(state)
    finally:
        handle.remove()
    expected = torch.stack(
        (state.load, state.current_time, state.length, state.open), dim=2
    )
    assert len(captured) == 1
    assert torch.equal(captured[0][:, :, -4:], expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("problem", TRAIN_PROBLEMS)
@pytest.mark.parametrize("problem_size", (50, 100))
def test_three_models_have_finite_forward_backward_at_both_sizes(
    model_class, problem, problem_size,
):
    from SplitTrainer import valid_pomo_reinforce_loss

    torch.manual_seed(123)
    source = official_env(problem, batch=1, n=problem_size, pomo=4)
    env = GiantTourEnv.from_official_env(source, pomo_size=4)
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
    for pomo_index in range(env.pomo_size):
        if not bool(env.last_split_result.feasible[0, pomo_index]):
            continue
        routes = env.get_routes(0, pomo_index)
        replay = verify_routes(
            env.depot_xy, env.node_xy, routes, env.instance.split_view
        )
        assert replay.feasible, replay.reason
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    input_size = getattr(
        model.encoder.embedding_node,
        "input_size",
        getattr(model.encoder.embedding_node, "in_features", None),
    )
    assert input_size == 5
    assert model.decoder.Wq_last.in_features == params()["embedding_dim"] + 4
    assert not hasattr(model.decoder, "Wq_first")
    assert not hasattr(model.decoder, "Wq_b")
    assert not hasattr(model.decoder, "b_candidate_score")
