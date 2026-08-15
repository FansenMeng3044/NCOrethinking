import os
import sys
import unittest

import torch


CVRP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEW_PY_DIR = os.path.abspath(os.path.join(CVRP_DIR, ".."))
AM_DIR = os.path.join(CVRP_DIR, "AM_SPLIT")
sys.path[:0] = [AM_DIR, CVRP_DIR, NEW_PY_DIR]

from CVRPTWCore import replay_cvrptw_actions, split_giant_tours_tw
from POMO_SPLIT.GiantTourModel import GiantTourModel
from POMO_TW.VRPTWEnv import VRPTWEnv
from POMO_TW.VRPTWModel import VRPTWModel
from nets.attention_model import AttentionModel, set_decode_type
from problems.am_split_tw.problem_am_split_tw import AMSplitTW
from problems.cvrptw.problem_cvrptw import CVRPTW


class SharedSemanticsTest(unittest.TestCase):
    def test_hard_tw_split_forces_two_independent_vehicle_routes(self):
        depot = torch.tensor([[[0.0, 0.0]]])
        nodes = torch.tensor([[[1.0, 0.0], [1.2, 0.0]]])
        demand = torch.tensor([[0.1, 0.1]])
        service = torch.zeros(1, 2)
        # Customer 1 makes the vehicle wait until t=2.  Customer 2 must start
        # by t=1.5, so [1, 2] is infeasible but the two single-customer routes
        # are both feasible because each fresh vehicle starts at t=0.
        tw_start = torch.tensor([[2.0, 0.0]])
        tw_end = torch.tensor([[3.0, 1.5]])
        tour = torch.tensor([[1, 2]])
        result = split_giant_tours_tw(
            depot,
            nodes,
            demand,
            service,
            tw_start,
            tw_end,
            tour,
            depot_end=5.0,
            return_predecessors=True,
        )
        self.assertAlmostEqual(result.costs.item(), 4.4, places=5)
        self.assertEqual(result.route_counts.item(), 2)
        self.assertEqual(result.predecessors[0].tolist(), [-1, 0, 1])

    def test_pomo_depot_resets_capacity_and_time_but_reward_is_total_distance(self):
        env = VRPTWEnv(
            problem_size=3,
            pomo_size=1,
            device="cpu",
            depot_end=10.0,
            service_duration=0.0,
        )
        problems = (
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
            torch.tensor([[0.6, 0.6, 0.4]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.full((1, 3), 10.0),
        )
        env.load_problems_manual(*problems)
        env.reset()
        for action in (0, 1):
            env.step(torch.tensor([[action]]))
        self.assertAlmostEqual(env.current_time.item(), 1.0, places=6)
        self.assertAlmostEqual(env.load.item(), 0.4, places=6)
        env.step(torch.tensor([[0]]))
        self.assertEqual(env.current_time.item(), 0.0)
        self.assertEqual(env.load.item(), 1.0)
        reward = None
        done = False
        for action in (2, 3, 0):
            _, reward, done = env.step(torch.tensor([[action]]))
        self.assertTrue(done)
        replay = env.strict_replay()
        self.assertTrue(replay.feasible.item())
        self.assertEqual(replay.route_counts.item(), 2)
        self.assertAlmostEqual((-reward).item(), replay.distances.item(), places=6)

    def test_replay_rejects_a_customer_time_window_violation(self):
        depot = torch.tensor([[[0.0, 0.0]]])
        nodes = torch.tensor([[[1.0, 0.0], [1.2, 0.0]]])
        result = replay_cvrptw_actions(
            depot,
            nodes,
            torch.tensor([[0.1, 0.1]]),
            torch.zeros(1, 2),
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[3.0, 1.5]]),
            torch.tensor([[1, 2, 0]]),
            depot_end=5.0,
        )
        self.assertTrue(result.all_customers_once.item())
        self.assertFalse(result.time_feasible.item())
        self.assertFalse(result.feasible.item())


class FourModelIntegrationTest(unittest.TestCase):
    MODEL_PARAMS = dict(
        embedding_dim=16,
        sqrt_embedding_dim=4.0,
        encoder_layer_num=2,
        qkv_dim=4,
        head_num=4,
        logit_clipping=10,
        ff_hidden_dim=32,
        eval_type="argmax",
    )

    def test_feature_contracts_are_full_tw_vs_xy_only(self):
        pomo_tw = VRPTWModel(**self.MODEL_PARAMS)
        pomo_split = GiantTourModel(**self.MODEL_PARAMS)
        am_tw = AttentionModel(
            16, 16, CVRPTW, n_encode_layers=2, n_heads=4, normalization="batch"
        )
        am_split = AttentionModel(
            16, 16, AMSplitTW, n_encode_layers=2, n_heads=4, normalization="batch"
        )
        self.assertEqual(pomo_tw.encoder.embedding_node.in_features, 6)
        self.assertEqual(pomo_split.encoder.embedding_node.in_features, 2)
        self.assertEqual(am_tw.init_embed.in_features, 6)
        self.assertEqual(am_split.init_embed.in_features, 2)

    def test_both_am_variants_decode_finite_training_batches(self):
        CVRPTW.configure()
        dataset = CVRPTW.make_dataset(size=20, num_samples=4)
        batch = {
            key: torch.stack([dataset[i][key] for i in range(4)])
            for key in dataset[0]
        }
        for problem in (CVRPTW, AMSplitTW):
            problem.configure()
            model = AttentionModel(
                16, 16, problem, n_encode_layers=2, n_heads=4,
                normalization="batch",
            )
            set_decode_type(model, "sampling")
            model.train()
            cost, log_likelihood = model(batch)
            loss = ((cost - cost.mean()).detach() * log_likelihood).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(cost).all().item())
            self.assertTrue(torch.isfinite(log_likelihood).all().item())
            self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
