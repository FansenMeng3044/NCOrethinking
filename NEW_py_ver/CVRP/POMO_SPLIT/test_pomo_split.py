import itertools
import math
import os
import sys
import unittest

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))

from GiantTourEnv import GiantTourEnv
from GiantTourModel import GiantTourModel
from SplitDecoder import reconstruct_routes, split_giant_tours


def brute_force_split(depot, nodes, demands, tour, capacity):
    best_cost = math.inf
    best_routes = None
    n = len(tour)
    for cut_bits in itertools.product((False, True), repeat=n - 1):
        routes = []
        begin = 0
        for pos, is_cut in enumerate(cut_bits, start=1):
            if is_cut:
                routes.append(tour[begin:pos])
                begin = pos
        routes.append(tour[begin:n])

        if any(sum(float(demands[node - 1]) for node in route) > capacity + 1e-7
               for route in routes):
            continue

        cost = 0.0
        for route in routes:
            previous = depot
            for customer in route:
                current = nodes[customer - 1]
                cost += torch.linalg.vector_norm(current - previous).item()
                previous = current
            cost += torch.linalg.vector_norm(previous - depot).item()
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
    return best_cost, best_routes


class SplitDecoderTest(unittest.TestCase):
    def test_known_optimal_split_and_reconstruction(self):
        depot = torch.tensor([[[0.0, 0.0]]])
        nodes = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        demands = torch.tensor([[0.6, 0.4, 0.6]])
        tours = torch.tensor([[[1, 2, 3]]])

        result = split_giant_tours(
            depot, nodes, demands, tours, capacity=1.0, return_predecessors=True
        )

        self.assertAlmostEqual(result.costs.item(), 8.0, places=6)
        self.assertEqual(
            reconstruct_routes(tours[0, 0], result.predecessors[0, 0]),
            [[1], [2, 3]],
        )

    def test_vectorized_split_matches_exhaustive_enumeration(self):
        generator = torch.Generator().manual_seed(1234)
        batch, pomo, n = 2, 4, 6
        depot = torch.rand(batch, 1, 2, generator=generator)
        nodes = torch.rand(batch, n, 2, generator=generator)
        demands = torch.randint(1, 5, (batch, n), generator=generator).float() / 10
        tours = torch.stack(
            [torch.stack([torch.randperm(n, generator=generator) + 1 for _ in range(pomo)])
             for _ in range(batch)]
        )

        result = split_giant_tours(
            depot, nodes, demands, tours, capacity=1.0, return_predecessors=True
        )

        for b in range(batch):
            for p in range(pomo):
                expected, _ = brute_force_split(
                    depot[b, 0], nodes[b], demands[b], tours[b, p].tolist(), 1.0
                )
                self.assertAlmostEqual(result.costs[b, p].item(), expected, places=5)
                routes = reconstruct_routes(tours[b, p], result.predecessors[b, p])
                flattened = [customer for route in routes for customer in route]
                self.assertEqual(flattened, tours[b, p].tolist())
                for route in routes:
                    load = sum(demands[b, customer - 1].item() for customer in route)
                    self.assertLessEqual(load, 1.0 + 1e-6)

    def test_rejects_non_permutation(self):
        depot = torch.zeros(1, 1, 2)
        nodes = torch.rand(1, 3, 2)
        demands = torch.full((1, 3), 0.2)
        bad_tour = torch.tensor([[[1, 1, 3]]])
        with self.assertRaisesRegex(ValueError, "permutation"):
            split_giant_tours(depot, nodes, demands, bad_tour)


class PomoGiantTourIntegrationTest(unittest.TestCase):
    def test_decoder_receives_direct_load_state_without_capacity_mask(self):
        env = GiantTourEnv(problem_size=2, pomo_size=1, capacity=1.0, device="cpu")
        env.load_problems_manual(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),
            torch.tensor([[0.8, 0.8]]),
        )
        state, _, _ = env.reset()
        model = GiantTourModel(
            node_feature_dim=3,
            embedding_dim=16,
            sqrt_embedding_dim=4.0,
            encoder_layer_num=1,
            qkv_dim=4,
            head_num=4,
            logit_clipping=10,
            ff_hidden_dim=32,
            eval_type="argmax",
        )
        self.assertEqual(model.decoder.Wq_last.in_features, 17)
        state, _, _ = env.step(torch.tensor([[1]]))
        self.assertAlmostEqual(state.load.item(), 0.2, places=6)
        self.assertFalse(state.ninf_mask[0, 0, 2].isneginf().item())
        state, _, done = env.step(torch.tensor([[2]]))
        self.assertTrue(done)
        self.assertAlmostEqual(state.load.item(), -0.6, places=6)

    def test_policy_selects_customers_only_and_reward_is_split_cost(self):
        torch.manual_seed(7)
        n = 5
        env = GiantTourEnv(problem_size=n, pomo_size=n, capacity=1.0, device="cpu")
        depot = torch.tensor([[[0.25, 0.25]]])
        nodes = torch.rand(1, n, 2)
        demands = torch.tensor([[0.6, 0.4, 0.6, 0.3, 0.5]])
        env.load_problems_manual(depot, nodes, demands)

        model = GiantTourModel(
            node_feature_dim=3,
            embedding_dim=16,
            sqrt_embedding_dim=4.0,
            encoder_layer_num=2,
            qkv_dim=4,
            head_num=4,
            logit_clipping=10,
            ff_hidden_dim=32,
            eval_type="argmax",
        )
        self.assertEqual(model.encoder.embedding_node.in_features, 3)
        model.train()
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        probabilities = []
        while not done:
            selected, prob = model(state)
            self.assertTrue((selected >= 1).all().item())
            self.assertTrue((selected <= n).all().item())
            probabilities.append(prob)
            state, reward, done = env.step(selected)

        sorted_tours = env.selected_node_list.sort(dim=2).values
        expected = torch.arange(1, n + 1).view(1, 1, n).expand_as(sorted_tours)
        self.assertTrue(torch.equal(sorted_tours, expected))
        self.assertTrue(torch.allclose(reward, -env.last_split_result.costs))

        for p in range(n):
            routes = env.get_routes(0, p)
            for route in routes:
                route_load = sum(demands[0, customer - 1].item() for customer in route)
                self.assertLessEqual(route_load, 1.0 + 1e-6)

        log_prob = torch.stack(probabilities, dim=2).log().sum(dim=2)
        reinforce_loss = -(reward.detach() * log_prob).mean()
        reinforce_loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
