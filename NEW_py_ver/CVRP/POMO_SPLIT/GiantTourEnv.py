from dataclasses import dataclass

import torch

try:
    from POMO.CVRProblemDef import augment_xy_data_by_8_fold, get_random_problems
except ImportError:  # Keep the original standalone-script entry points working.
    from CVRProblemDef import augment_xy_data_by_8_fold, get_random_problems
try:
    from .SplitDecoder import reconstruct_routes, split_giant_tours
except ImportError:  # Keep the original standalone-script entry points working.
    from SplitDecoder import reconstruct_routes, split_giant_tours


@dataclass
class ResetState:
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class GiantTourEnv:
    """CVRP environment whose policy constructs only a customer giant tour."""

    def __init__(self, **env_params):
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        self.capacity = env_params.get("capacity", 1.0)
        self.device = torch.device(env_params.get("device", "cpu"))
        if self.pomo_size > self.problem_size:
            raise ValueError("pomo_size cannot exceed problem_size")

        # Optional fixed-test-set support (mirrors CVRPEnv.use_saved_problems).
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = 0

        self.batch_size = None
        self.depot_xy = None
        self.node_xy = None
        self.node_demand = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.step_state = None
        self.last_split_result = None

    def use_saved_problems(self, filename, device=None):
        """Load a fixed CVRP test set so evaluation matches the official runs.

        The file must be the same format used by CVRPEnv.use_saved_problems: a
        dict with depot_xy (N,1,2), node_xy (N,n,2) and node_demand (N,n), where
        demands are already normalized to the vehicle capacity.
        """
        map_location = device if device is not None else self.device
        loaded_dict = torch.load(filename, map_location=map_location)
        self.saved_depot_xy = loaded_dict["depot_xy"]
        self.saved_node_xy = loaded_dict["node_xy"]
        self.saved_node_demand = loaded_dict["node_demand"]
        self.saved_index = 0
        self.FLAG__use_saved_problems = True

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            end = self.saved_index + batch_size
            depot_xy = self.saved_depot_xy[self.saved_index:end]
            node_xy = self.saved_node_xy[self.saved_index:end]
            node_demand = self.saved_node_demand[self.saved_index:end]
            self.saved_index = end
        depot_xy = depot_xy.to(self.device)
        node_xy = node_xy.to(self.device)
        node_demand = node_demand.to(self.device)

        if aug_factor == 8:
            self.batch_size *= 8
            depot_xy = augment_xy_data_by_8_fold(depot_xy)
            node_xy = augment_xy_data_by_8_fold(node_xy)
            node_demand = node_demand.repeat(8, 1)
        elif aug_factor != 1:
            raise NotImplementedError("only augmentation factors 1 and 8 are supported")

        self._set_problems(depot_xy, node_xy, node_demand)

    def load_problems_manual(self, depot_xy, node_xy, node_demand):
        """Load explicit tensors, primarily for fixed evaluation and tests."""
        self.batch_size = depot_xy.size(0)
        if node_xy.size(1) != self.problem_size:
            raise ValueError("manual problem size does not match environment problem_size")
        self._set_problems(
            depot_xy.to(self.device), node_xy.to(self.device), node_demand.to(self.device)
        )

    def _set_problems(self, depot_xy, node_xy, node_demand):
        self.depot_xy = depot_xy
        self.node_xy = node_xy
        self.node_demand = node_demand
        device = node_xy.device
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(
            self.batch_size, self.pomo_size
        )

    def reset(self):
        device = self.node_xy.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            self.batch_size, self.pomo_size, 0, device=device, dtype=torch.long
        )
        self.step_state = StepState(self.BATCH_IDX, self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            self.batch_size,
            self.pomo_size,
            self.problem_size + 1,
            device=device,
            dtype=self.node_xy.dtype,
        )
        # Depot is encoder context only and can never be selected as an action.
        self.step_state.ninf_mask[:, :, 0] = float("-inf")
        self.last_split_result = None
        return ResetState(self.depot_xy, self.node_xy, self.node_demand), None, False

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2
        )
        self.step_state.current_node = selected
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")

        done = self.selected_count == self.problem_size
        if not done:
            return self.step_state, None, False

        with torch.no_grad():
            self.last_split_result = split_giant_tours(
                self.depot_xy,
                self.node_xy,
                self.node_demand,
                self.selected_node_list,
                capacity=self.capacity,
                return_predecessors=True,
            )
        reward = -self.last_split_result.costs
        return self.step_state, reward, True

    def get_routes(self, batch_index, pomo_index):
        if self.last_split_result is None:
            raise RuntimeError("a rollout must finish before routes can be reconstructed")
        return reconstruct_routes(
            self.selected_node_list[batch_index, pomo_index],
            self.last_split_result.predecessors[batch_index, pomo_index],
        )
