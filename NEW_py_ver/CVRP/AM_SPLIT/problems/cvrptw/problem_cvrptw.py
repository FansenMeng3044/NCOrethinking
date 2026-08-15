import os
import pickle

import torch
from torch.utils.data import Dataset

from CVRPTWCore import get_random_problems, replay_cvrptw_actions, tensors_from_dict
from problems.cvrptw.state_cvrptw import StateCVRPTW
from utils.beam_search import beam_search


class CVRPTW:
    NAME = "cvrptw"
    VEHICLE_CAPACITY = 1.0
    DEPOT_START = 0.0
    DEPOT_END = 3.0
    SPEED = 1.0
    SERVICE_DURATION = 0.2

    @classmethod
    def configure(
        cls, capacity=1.0, depot_start=0.0, depot_end=3.0,
        speed=1.0, service_duration=0.2, **_unused
    ):
        cls.VEHICLE_CAPACITY = float(capacity)
        cls.DEPOT_START = float(depot_start)
        cls.DEPOT_END = float(depot_end)
        cls.SPEED = float(speed)
        cls.SERVICE_DURATION = float(service_duration)
        StateCVRPTW.VEHICLE_CAPACITY = cls.VEHICLE_CAPACITY

    @classmethod
    def get_costs(cls, dataset, pi):
        replay = replay_cvrptw_actions(
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
        )
        if not replay.feasible.all():
            raise ValueError("Attention Model produced an infeasible CVRPTW solution")
        return replay.distances, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)

    @classmethod
    def beam_search(
        cls, input, beam_size, expand_size=None, compress_mask=False,
        model=None, max_calc_batch_size=4096
    ):
        if model is None:
            raise ValueError("beam_search requires a model")
        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = cls.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )
        return beam_search(state, beam_size, propose_expansions)


class CVRPTWDataset(Dataset):
    def __init__(
        self, filename=None, size=100, num_samples=1000000, offset=0,
        distribution=None
    ):
        if distribution is not None:
            raise ValueError("CVRPTW supports the uniform distribution only")
        if filename is None:
            values = get_random_problems(
                num_samples,
                size,
                device="cpu",
                depot_start=CVRPTW.DEPOT_START,
                depot_end=CVRPTW.DEPOT_END,
                speed=CVRPTW.SPEED,
                service_duration=CVRPTW.SERVICE_DURATION,
            )
        else:
            values = self._load_file(filename)
            end = min(offset + num_samples, values[0].size(0))
            values = tuple(value[offset:end] for value in values)
        (
            self.depot, self.loc, self.demand, self.service_time,
            self.tw_start, self.tw_end,
        ) = values
        self.depot = self.depot[:, 0]

    @staticmethod
    def _load_file(filename):
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".pt":
            try:
                data = torch.load(filename, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(filename, map_location="cpu")
            return tensors_from_dict(data, device="cpu")
        if extension == ".pkl":
            with open(filename, "rb") as handle:
                rows = pickle.load(handle)
            depots, locs, demands, services, starts, ends = [], [], [], [], [], []
            for row in rows:
                depot, loc, demand, capacity, service, tw_start, tw_end = row[:7]
                depots.append(torch.as_tensor(depot, dtype=torch.float).reshape(1, 2))
                locs.append(torch.as_tensor(loc, dtype=torch.float))
                demands.append(torch.as_tensor(demand, dtype=torch.float) / float(capacity))
                services.append(torch.as_tensor(service, dtype=torch.float))
                starts.append(torch.as_tensor(tw_start, dtype=torch.float))
                ends.append(torch.as_tensor(tw_end, dtype=torch.float))
            return tuple(map(torch.stack, (depots, locs, demands, services, starts, ends)))
        raise ValueError("CVRPTW fixed datasets must be .pt or .pkl files")

    def __len__(self):
        return self.loc.size(0)

    def __getitem__(self, index):
        return {
            "depot": self.depot[index],
            "loc": self.loc[index],
            "demand": self.demand[index],
            "service_time": self.service_time[index],
            "tw_start": self.tw_start[index],
            "tw_end": self.tw_end[index],
            "depot_start": self.loc.new_tensor(CVRPTW.DEPOT_START),
            "depot_end": self.loc.new_tensor(CVRPTW.DEPOT_END),
            "speed": self.loc.new_tensor(CVRPTW.SPEED),
        }
