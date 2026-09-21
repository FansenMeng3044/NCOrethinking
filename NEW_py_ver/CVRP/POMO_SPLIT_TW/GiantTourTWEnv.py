from dataclasses import dataclass

import torch

from CVRPTWCore import (
    VRPTW_CAPACITY,
    VRPTW_DEPOT_END,
    VRPTW_DEPOT_START,
    VRPTW_EPSILON,
    VRPTW_SERVICE_DURATION,
    VRPTW_SPEED,
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
    load: torch.Tensor = None
    current_time: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class GiantTourTWEnv:
    """POMO customer ordering followed by exact hard CVRPTW Split."""

    def __init__(self, **env_params):
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        if self.pomo_size > self.problem_size:
            raise ValueError("pomo_size cannot exceed problem_size")
        self.capacity = float(env_params.get("capacity", VRPTW_CAPACITY))
        self.speed = float(env_params.get("speed", VRPTW_SPEED))
        self.depot_start = float(env_params.get("depot_start", VRPTW_DEPOT_START))
        self.depot_end = float(env_params.get("depot_end", VRPTW_DEPOT_END))
        self.service_duration = float(
            env_params.get("service_duration", VRPTW_SERVICE_DURATION)
        )
        self.epsilon = float(env_params.get("epsilon", VRPTW_EPSILON))
        self.loc_scaler = env_params.get("loc_scaler")
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
        shape = (self.batch_size, self.pomo_size)
        self.load = torch.full(
            shape, self.capacity, device=self.device, dtype=self.node_demand.dtype
        )
        self.current_time = torch.full(
            shape, self.depot_start, device=self.device, dtype=self.node_xy.dtype
        )
        self.current_coord = self.depot_xy.expand(-1, self.pomo_size, -1).clone()
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
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
        selected_customer = selected - 1
        selected_coord = self.node_xy[self.BATCH_IDX, selected_customer]
        selected_demand = self.node_demand[self.BATCH_IDX, selected_customer]
        selected_service = self.node_service_time[
            self.BATCH_IDX, selected_customer
        ]
        selected_tw_start = self.node_tw_start[
            self.BATCH_IDX, selected_customer
        ]
        travel = torch.linalg.vector_norm(
            selected_coord - self.current_coord, dim=2
        )
        service_start = torch.maximum(
            self.current_time + travel / self.speed, selected_tw_start
        )
        # These are the same dynamic quantities supplied to the Direct
        # decoder. Capacity and time-window violations do not enter the mask;
        # the final Split decoder remains responsible for feasibility.
        self.load = self.load - selected_demand
        self.current_time = service_start + selected_service
        self.current_coord = selected_coord
        self.step_state.current_node = selected
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
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
                loc_scaler=self.loc_scaler,
                epsilon=self.epsilon,
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
