import os
import pickle

import torch
from torch.utils.data import Dataset

from problems.am_split.state_am_split import StateAMSplit
from split_decoder import raw_giant_tour_cost, split_giant_tours


class AMSplit:
    """Demand-blind Attention Model followed by exact capacity Split."""

    NAME = "am_split"
    VEHICLE_CAPACITY = 1.0
    TRAIN_REWARD = "split"

    @classmethod
    def configure(cls, capacity=1.0, train_reward="split"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if train_reward not in ("split", "raw"):
            raise ValueError("train_reward must be 'split' or 'raw'")
        cls.VEHICLE_CAPACITY = float(capacity)
        cls.TRAIN_REWARD = train_reward

    @classmethod
    def get_costs(cls, dataset, pi):
        cls._validate_tours(dataset, pi)
        if cls.TRAIN_REWARD == "raw":
            return raw_giant_tour_cost(dataset["depot"], dataset["loc"], pi), None
        with torch.no_grad():
            result = split_giant_tours(
                dataset["depot"],
                dataset["loc"],
                dataset["demand"],
                pi,
                capacity=cls.VEHICLE_CAPACITY,
            )
        return result.costs, None

    @classmethod
    def split_costs(cls, dataset, pi, return_predecessors=False):
        cls._validate_tours(dataset, pi)
        return split_giant_tours(
            dataset["depot"],
            dataset["loc"],
            dataset["demand"],
            pi,
            capacity=cls.VEHICLE_CAPACITY,
            return_predecessors=return_predecessors,
        )

    @staticmethod
    def _validate_tours(dataset, pi):
        n = dataset["loc"].size(1)
        expected = torch.arange(1, n + 1, device=pi.device, dtype=pi.dtype)
        if pi.shape != (dataset["loc"].size(0), n):
            raise ValueError("AM Split must output one length-n tour per instance")
        if not torch.equal(pi.sort(dim=1).values, expected[None].expand_as(pi)):
            raise ValueError("AM Split output is not a customer permutation")

    @staticmethod
    def make_dataset(*args, **kwargs):
        return AMSplitDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateAMSplit.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(*args, **kwargs):
        raise NotImplementedError("AM Split evaluation supports greedy and sampling decoding")


class AMSplitDataset(Dataset):
    """Random CVRP data or POMO .pt / original AM .pkl fixed datasets."""

    DEMAND_SCALERS = {20: 30.0, 50: 40.0, 100: 50.0}

    def __init__(
        self,
        filename=None,
        size=100,
        num_samples=1000000,
        offset=0,
        distribution=None,
    ):
        super().__init__()
        if distribution is not None:
            raise ValueError("AM Split currently supports the uniform CVRP distribution only")

        if filename is None:
            if size not in self.DEMAND_SCALERS:
                raise NotImplementedError("random instances support graph sizes 20, 50 and 100")
            self.depot = torch.rand(num_samples, 2)
            self.loc = torch.rand(num_samples, size, 2)
            self.demand = (
                torch.randint(1, 10, (num_samples, size)).float()
                / self.DEMAND_SCALERS[size]
            )
        else:
            self.depot, self.loc, self.demand = self._load_file(filename)
            end = min(offset + num_samples, self.loc.size(0))
            self.depot = self.depot[offset:end]
            self.loc = self.loc[offset:end]
            self.demand = self.demand[offset:end]

        if not (self.depot.size(0) == self.loc.size(0) == self.demand.size(0)):
            raise ValueError("dataset tensors have inconsistent sample counts")
        if self.depot.shape != (self.loc.size(0), 2):
            raise ValueError("depot must have shape (samples, 2)")
        if self.demand.shape != self.loc.shape[:2]:
            raise ValueError("demand must have shape (samples, graph_size)")

    @staticmethod
    def _load_file(filename):
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".pt":
            try:
                data = torch.load(filename, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(filename, map_location="cpu")
            depot = data["depot_xy"].float()
            if depot.dim() == 3 and depot.size(1) == 1:
                depot = depot[:, 0]
            return depot, data["node_xy"].float(), data["node_demand"].float()
        if extension == ".pkl":
            with open(filename, "rb") as handle:
                rows = pickle.load(handle)
            depots, locs, demands = [], [], []
            for row in rows:
                depot, loc, demand, capacity = row[:4]
                depots.append(torch.tensor(depot, dtype=torch.float))
                locs.append(torch.tensor(loc, dtype=torch.float))
                demands.append(torch.tensor(demand, dtype=torch.float) / float(capacity))
            return torch.stack(depots), torch.stack(locs), torch.stack(demands)
        raise ValueError("fixed datasets must be .pt or .pkl files")

    def __len__(self):
        return self.loc.size(0)

    def __getitem__(self, index):
        return {
            "depot": self.depot[index],
            "loc": self.loc[index],
            "demand": self.demand[index],
        }
