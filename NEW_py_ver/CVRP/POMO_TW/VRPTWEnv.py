from dataclasses import dataclass
from pathlib import Path

import torch

from CVRPTWCore import (
    augment_problems_by_8,
    get_random_problems,
    replay_cvrptw_actions,
    tensors_from_dict,
    validate_problem_tensors,
)


@dataclass
class ResetState:
    depot_xy: torch.Tensor = None
    node_xy: torch.Tensor = None
    node_demand: torch.Tensor = None
    node_service_time: torch.Tensor = None
    node_tw_start: torch.Tensor = None
    node_tw_end: torch.Tensor = None


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    load: torch.Tensor = None
    current_time: torch.Tensor = None
    route_length: torch.Tensor = None
    route_count: torch.Tensor = None
    current_coord: torch.Tensor = None


class VRPTWEnv:
    """POMO environment for homogeneous-fleet multi-vehicle CVRPTW.

    A depot action closes the current vehicle route.  Capacity is refilled and
    time is reset to ``depot_start`` for a fresh vehicle.  Reward is negative
    total Euclidean distance across every depot-delimited route.
    """

    def __init__(self, **env_params):
        self.env_params = dict(env_params)
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        if self.pomo_size > self.problem_size:
            raise ValueError("pomo_size cannot exceed problem_size")
        self.capacity = float(env_params.get("capacity", 1.0))
        self.speed = float(env_params.get("speed", 1.0))
        self.depot_start = float(env_params.get("depot_start", 0.0))
        self.depot_end = float(env_params.get("depot_end", 3.0))
        self.service_duration = float(env_params.get("service_duration", 0.2))
        self.epsilon = float(env_params.get("epsilon", 1e-5))
        self.device = torch.device(env_params.get("device", "cpu"))

        self.saved_problems = None
        self.saved_index = 0
        self.batch_size = None
        self.reset_state = ResetState()
        self.step_state = StepState()

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
            selected = tuple(t.to(self.device) for t in problems)
        elif self.saved_problems is not None:
            end = self.saved_index + batch_size
            if end > self.saved_problems[0].size(0):
                raise IndexError("fixed CVRPTW dataset is exhausted")
            selected = tuple(t[self.saved_index:end].to(self.device) for t in self.saved_problems)
            self.saved_index = end
        else:
            selected = get_random_problems(
                batch_size,
                self.problem_size,
                device=self.device,
                depot_start=self.depot_start,
                depot_end=self.depot_end,
                speed=self.speed,
                service_duration=self.service_duration,
            )
        validate_problem_tensors(*selected)
        if selected[1].size(1) != self.problem_size:
            raise ValueError("loaded problem size does not match the environment")
        if aug_factor == 8:
            selected = augment_problems_by_8(selected)
        elif aug_factor != 1:
            raise NotImplementedError("only augmentation factors 1 and 8 are supported")
        self._set_problems(*selected)

    def load_problems_manual(self, *problems):
        self.load_problems(problems[0].size(0), problems=problems)

    def _set_problems(
        self, depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
    ):
        self.batch_size = depot_xy.size(0)
        self.depot_xy = depot_xy
        self.node_xy = node_xy
        self.node_demand = node_demand
        self.node_service_time = service_time
        self.node_tw_start = tw_start
        self.node_tw_end = tw_end
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        zeros = node_demand.new_zeros(self.batch_size, 1)
        self.depot_node_demand = torch.cat((zeros, node_demand), dim=1)
        self.depot_node_service_time = torch.cat((zeros, service_time), dim=1)
        self.depot_node_tw_start = torch.cat(
            (zeros + self.depot_start, tw_start), dim=1
        )
        self.depot_node_tw_end = torch.cat((zeros + self.depot_end, tw_end), dim=1)
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(
            self.batch_size, self.pomo_size
        )
        self.reset_state = ResetState(
            depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
        )
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(
            1, self.pomo_size + 1, device=self.device
        )[None].expand(self.batch_size, -1)

    def reset(self):
        if self.batch_size is None:
            raise RuntimeError("load_problems must be called before reset")
        shape = (self.batch_size, self.pomo_size)
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            *shape, 0, dtype=torch.long, device=self.device
        )
        self.at_the_depot = torch.ones(shape, dtype=torch.bool, device=self.device)
        self.load = torch.full(shape, self.capacity, device=self.device)
        self.visited_ninf_flag = torch.zeros(
            *shape, self.problem_size + 1, device=self.device
        )
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.finished = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.current_time = torch.full(shape, self.depot_start, device=self.device)
        self.route_length = torch.zeros(shape, device=self.device)
        self.route_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.current_coord = self.depot_xy.expand(-1, self.pomo_size, -1).clone()
        self._update_step_state()
        return self.reset_state, None, False

    def pre_step(self):
        self._update_step_state()
        return self.step_state, None, False

    def step(self, selected):
        if selected.shape != (self.batch_size, self.pomo_size):
            raise ValueError("selected must have shape (batch, pomo)")
        if self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected].isneginf().any():
            raise ValueError("decoder selected an infeasible CVRPTW action")

        was_at_depot = self.at_the_depot
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2
        )
        self.at_the_depot = selected == 0
        self.route_count += ((~self.at_the_depot) & was_at_depot).long()

        selected_coord = self.depot_node_xy[self.BATCH_IDX, selected]
        travel_distance = (selected_coord - self.current_coord).norm(p=2, dim=-1)
        self.route_length += travel_distance
        self.current_coord = selected_coord

        selected_demand = self.depot_node_demand[self.BATCH_IDX, selected]
        self.load -= selected_demand
        self.load[self.at_the_depot] = self.capacity

        selected_tw_start = self.depot_node_tw_start[self.BATCH_IDX, selected]
        selected_service = self.depot_node_service_time[self.BATCH_IDX, selected]
        service_start = torch.maximum(
            self.current_time + travel_distance / self.speed, selected_tw_start
        )
        self.current_time = service_start + selected_service
        self.current_time[self.at_the_depot] = self.depot_start
        self.route_length[self.at_the_depot] = 0

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0
        self.ninf_mask = self.visited_ninf_flag.clone()

        demand = self.depot_node_demand[:, None, :]
        self.ninf_mask[
            self.load[:, :, None] + self.epsilon < demand
        ] = float("-inf")

        candidate_distance = (
            self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :]
        ).norm(p=2, dim=-1)
        candidate_service_start = torch.maximum(
            self.current_time[:, :, None] + candidate_distance / self.speed,
            self.depot_node_tw_start[:, None, :],
        )
        out_of_window = (
            candidate_service_start
            > self.depot_node_tw_end[:, None, :] + self.epsilon
        )
        return_distance = (
            self.depot_node_xy[:, :, :] - self.depot_xy
        ).norm(p=2, dim=-1)
        cannot_return = (
            candidate_service_start
            + self.depot_node_service_time[:, None, :]
            + return_distance[:, None, :] / self.speed
            > self.depot_end + self.epsilon
        )
        self.ninf_mask[out_of_window | cannot_return] = float("-inf")

        newly_finished = self.visited_ninf_flag.isneginf().all(dim=2)
        self.finished |= newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0
        self._update_step_state()
        done = bool(self.finished.all())
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
        self.step_state.route_length = self.route_length
        self.step_state.route_count = self.route_count
        self.step_state.current_coord = self.current_coord

    def _get_travel_distance(self):
        gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None].expand(-1, self.pomo_size, -1, -1)
        ordered = all_xy.gather(2, gather)
        return (ordered - ordered.roll(shifts=-1, dims=2)).norm(p=2, dim=-1).sum(2)

    def strict_replay(self):
        if not bool(self.finished.all()):
            raise RuntimeError("rollout must finish before strict replay")
        route_batch = self.batch_size * self.pomo_size
        actions = self.selected_node_list.reshape(route_batch, -1)
        expand = lambda t: t[:, None].expand(-1, self.pomo_size, *([-1] * (t.dim() - 1))).reshape(
            route_batch, *t.shape[1:]
        )
        return replay_cvrptw_actions(
            expand(self.depot_xy),
            expand(self.node_xy),
            expand(self.node_demand),
            expand(self.node_service_time),
            expand(self.node_tw_start),
            expand(self.node_tw_end),
            actions,
            capacity=self.capacity,
            depot_start=self.depot_start,
            depot_end=self.depot_end,
            speed=self.speed,
        )

    @staticmethod
    def save_dataset(path, problems):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = (
            "depot_xy", "node_xy", "node_demand", "node_service_time",
            "node_tw_start", "node_tw_end",
        )
        torch.save({name: tensor.detach().cpu() for name, tensor in zip(names, problems)}, path)
