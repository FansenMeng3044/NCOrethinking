from dataclasses import dataclass

import torch

from CVRPTWCore import (
    augment_problems_by_8,
    get_random_problems,
    reconstruct_routes,
    split_giant_tours_tw,
    tensors_from_dict,
    validate_problem_tensors,
)


@dataclass
class ResetState:
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor
    node_service_time: torch.Tensor
    node_tw_start: torch.Tensor
    node_tw_end: torch.Tensor


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class GiantTourTWEnv:
    """XY-only POMO giant tour followed by exact hard CVRPTW Split."""

    def __init__(self, **env_params):
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        if self.pomo_size > self.problem_size:
            raise ValueError("pomo_size cannot exceed problem_size")
        self.capacity = float(env_params.get("capacity", 1.0))
        self.speed = float(env_params.get("speed", 1.0))
        self.depot_start = float(env_params.get("depot_start", 0.0))
        self.depot_end = float(env_params.get("depot_end", 3.0))
        self.service_duration = float(env_params.get("service_duration", 0.2))
        self.device = torch.device(env_params.get("device", "cpu"))
        self.saved_problems = None
        self.saved_index = 0

    def use_saved_problems(self, filename, device=None):
        map_location = device if device is not None else self.device
        try:
            data = torch.load(filename, map_location=map_location, weights_only=False)
        except TypeError:
            data = torch.load(filename, map_location=map_location)
        self.saved_problems = tensors_from_dict(data, device=map_location)
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1, problems=None):
        if problems is not None:
            values = tuple(t.to(self.device) for t in problems)
        elif self.saved_problems is not None:
            end = self.saved_index + batch_size
            if end > self.saved_problems[0].size(0):
                raise IndexError("fixed CVRPTW dataset is exhausted")
            values = tuple(t[self.saved_index:end].to(self.device) for t in self.saved_problems)
            self.saved_index = end
        else:
            values = get_random_problems(
                batch_size,
                self.problem_size,
                device=self.device,
                depot_start=self.depot_start,
                depot_end=self.depot_end,
                speed=self.speed,
                service_duration=self.service_duration,
            )
        validate_problem_tensors(*values)
        if aug_factor == 8:
            values = augment_problems_by_8(values)
        elif aug_factor != 1:
            raise NotImplementedError("only augmentation factors 1 and 8 are supported")
        self._set_problems(*values)

    def load_problems_manual(self, *problems):
        self.load_problems(problems[0].size(0), problems=problems)

    def _set_problems(self, depot, loc, demand, service, tw_start, tw_end):
        self.batch_size = depot.size(0)
        if loc.size(1) != self.problem_size:
            raise ValueError("loaded problem size does not match environment")
        self.depot_xy, self.node_xy, self.node_demand = depot, loc, demand
        self.node_service_time, self.node_tw_start, self.node_tw_end = (
            service, tw_start, tw_end
        )
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None].expand(
            self.batch_size, self.pomo_size
        )

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            self.batch_size, self.pomo_size, 0, dtype=torch.long, device=self.device
        )
        self.step_state = StepState(self.BATCH_IDX, self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1, device=self.device
        )
        self.step_state.ninf_mask[:, :, 0] = float("-inf")
        self.last_split_result = None
        reset = ResetState(
            self.depot_xy, self.node_xy, self.node_demand,
            self.node_service_time, self.node_tw_start, self.node_tw_end,
        )
        return reset, None, False

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        if self.step_state.ninf_mask[
            self.BATCH_IDX, self.POMO_IDX, selected
        ].isneginf().any():
            raise ValueError("giant-tour policy selected an invalid customer")
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2
        )
        self.step_state.current_node = selected
        self.step_state.ninf_mask[
            self.BATCH_IDX, self.POMO_IDX, selected
        ] = float("-inf")
        if self.selected_count != self.problem_size:
            return self.step_state, None, False
        with torch.no_grad():
            self.last_split_result = split_giant_tours_tw(
                self.depot_xy,
                self.node_xy,
                self.node_demand,
                self.node_service_time,
                self.node_tw_start,
                self.node_tw_end,
                self.selected_node_list,
                capacity=self.capacity,
                depot_start=self.depot_start,
                depot_end=self.depot_end,
                speed=self.speed,
                return_predecessors=True,
            )
        return self.step_state, -self.last_split_result.costs, True

    def get_routes(self, batch_index, pomo_index):
        if self.last_split_result is None:
            raise RuntimeError("rollout must finish before route reconstruction")
        return reconstruct_routes(
            self.selected_node_list[batch_index, pomo_index],
            self.last_split_result.predecessors[batch_index, pomo_index],
        )
