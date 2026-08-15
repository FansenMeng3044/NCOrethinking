import torch

from CVRPTWCore import split_giant_tours_tw
from problems.am_split.state_am_split import StateAMSplit
from problems.cvrptw.problem_cvrptw import CVRPTW, CVRPTWDataset
from split_decoder import raw_giant_tour_cost


class AMSplitTW:
    """XY-only Attention Model followed by exact capacity-and-TW Split."""

    NAME = "am_split_tw"
    VEHICLE_CAPACITY = 1.0
    DEPOT_START = 0.0
    DEPOT_END = 3.0
    SPEED = 1.0
    SERVICE_DURATION = 0.2
    TRAIN_REWARD = "split"

    @classmethod
    def configure(
        cls, capacity=1.0, depot_start=0.0, depot_end=3.0,
        speed=1.0, service_duration=0.2, train_reward="split", **_unused
    ):
        if train_reward not in ("split", "raw"):
            raise ValueError("train_reward must be 'split' or 'raw'")
        cls.VEHICLE_CAPACITY = float(capacity)
        cls.DEPOT_START = float(depot_start)
        cls.DEPOT_END = float(depot_end)
        cls.SPEED = float(speed)
        cls.SERVICE_DURATION = float(service_duration)
        cls.TRAIN_REWARD = train_reward
        CVRPTW.configure(
            capacity=capacity,
            depot_start=depot_start,
            depot_end=depot_end,
            speed=speed,
            service_duration=service_duration,
        )

    @classmethod
    def get_costs(cls, dataset, pi):
        cls._validate_tours(dataset, pi)
        if cls.TRAIN_REWARD == "raw":
            return raw_giant_tour_cost(dataset["depot"], dataset["loc"], pi), None
        with torch.no_grad():
            result = cls.split_costs(dataset, pi)
        return result.costs, None

    @classmethod
    def split_costs(cls, dataset, pi, return_predecessors=False):
        cls._validate_tours(dataset, pi)
        return split_giant_tours_tw(
            dataset["depot"][:, None],
            dataset["loc"],
            dataset["demand"],
            dataset["service_time"],
            dataset["tw_start"],
            dataset["tw_end"],
            pi,
            capacity=cls.VEHICLE_CAPACITY,
            depot_start=dataset.get("depot_start", cls.DEPOT_START),
            depot_end=dataset.get("depot_end", cls.DEPOT_END),
            speed=cls.SPEED,
            return_predecessors=return_predecessors,
        )

    @staticmethod
    def _validate_tours(dataset, pi):
        n = dataset["loc"].size(1)
        if pi.shape != (dataset["loc"].size(0), n):
            raise ValueError("AM-Split-TW must output one length-n tour per instance")
        expected = torch.arange(1, n + 1, device=pi.device, dtype=pi.dtype)
        if not torch.equal(pi.sort(dim=1).values, expected[None].expand_as(pi)):
            raise ValueError("AM-Split-TW output is not a customer permutation")

    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateAMSplit.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(*args, **kwargs):
        raise NotImplementedError("AM-Split-TW supports greedy and sampling decoding")
