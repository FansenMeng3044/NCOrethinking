import itertools
import math
import unittest

import torch

from nets.attention_model import AttentionModel, set_decode_type
from problems.am_split.problem_am_split import AMSplit
from split_decoder import raw_giant_tour_cost, reconstruct_routes, split_giant_tours


def brute_force_split(depot, nodes, demands, tour, capacity):
    best = math.inf
    for cuts in itertools.product((False, True), repeat=len(tour) - 1):
        routes = []
        begin = 0
        for end, cut in enumerate(cuts, start=1):
            if cut:
                routes.append(tour[begin:end])
                begin = end
        routes.append(tour[begin:])
        if any(sum(demands[node - 1].item() for node in route) > capacity + 1e-7 for route in routes):
            continue
        cost = 0.0
        for route in routes:
            previous = depot
            for customer in route:
                current = nodes[customer - 1]
                cost += (current - previous).norm().item()
                previous = current
            cost += (previous - depot).norm().item()
        best = min(best, cost)
    return best


class SplitDecoderTest(unittest.TestCase):
    def test_known_split_and_reconstruction(self):
        depot = torch.tensor([[0.0, 0.0]])
        nodes = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        demands = torch.tensor([[0.6, 0.4, 0.6]])
        tour = torch.tensor([[1, 2, 3]])
        result = split_giant_tours(
            depot, nodes, demands, tour, return_predecessors=True
        )
        self.assertAlmostEqual(result.costs.item(), 8.0, places=6)
        self.assertEqual(reconstruct_routes(tour[0], result.predecessors[0]), [[1], [2, 3]])
        self.assertAlmostEqual(raw_giant_tour_cost(depot, nodes, tour).item(), 6.0, places=6)

    def test_vectorized_split_matches_exhaustive(self):
        generator = torch.Generator().manual_seed(1234)
        batch, n = 5, 6
        depot = torch.rand(batch, 2, generator=generator)
        nodes = torch.rand(batch, n, 2, generator=generator)
        demands = torch.randint(1, 5, (batch, n), generator=generator).float() / 10
        tours = torch.stack([torch.randperm(n, generator=generator) + 1 for _ in range(batch)])
        actual = split_giant_tours(depot, nodes, demands, tours).costs
        for index in range(batch):
            expected = brute_force_split(
                depot[index], nodes[index], demands[index], tours[index].tolist(), 1.0
            )
            self.assertAlmostEqual(actual[index].item(), expected, places=5)


class AttentionModelIntegrationTest(unittest.TestCase):
    def make_model(self):
        return AttentionModel(
            embedding_dim=16,
            hidden_dim=16,
            problem=AMSplit,
            n_encode_layers=1,
            tanh_clipping=10.0,
            normalization="instance",
            n_heads=4,
        )

    def test_customer_only_permutation_and_backward(self):
        torch.manual_seed(7)
        AMSplit.configure(capacity=1.0, train_reward="split")
        batch, n = 3, 5
        data = {
            "depot": torch.rand(batch, 2),
            "loc": torch.rand(batch, n, 2),
            "demand": torch.randint(1, 5, (batch, n)).float() / 10,
        }
        model = self.make_model()
        model.train()
        set_decode_type(model, "sampling")
        cost, ll, tour = model(data, return_pi=True)
        expected = torch.arange(1, n + 1)[None].expand_as(tour)
        self.assertTrue(torch.equal(tour.sort(dim=1).values, expected))
        self.assertTrue(torch.allclose(cost, AMSplit.split_costs(data, tour).costs))
        (cost.detach() * ll).mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))

    def test_policy_is_demand_blind(self):
        torch.manual_seed(11)
        AMSplit.configure(capacity=1.0, train_reward="split")
        n = 5
        depot = torch.rand(1, 2).repeat(2, 1)
        loc = torch.rand(1, n, 2).repeat(2, 1, 1)
        demand = torch.stack((torch.full((n,), 0.1), torch.full((n,), 0.4)))
        model = self.make_model()
        model.eval()
        set_decode_type(model, "greedy")
        _, _, tours = model(
            {"depot": depot, "loc": loc, "demand": demand}, return_pi=True
        )
        self.assertTrue(torch.equal(tours[0], tours[1]))


if __name__ == "__main__":
    unittest.main()
