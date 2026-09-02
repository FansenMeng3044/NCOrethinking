from dataclasses import dataclass
from typing import Optional

import torch

from split.decoder import SplitResult, reconstruct_routes, split_giant_tours
from .adapter import AdaptedInstance, MVMoEInstanceAdapter


@dataclass
class PolicyResetState:
    # Intentionally no demand, time-window, route-limit, problem token, or flag.
    depot_xy: torch.Tensor
    node_xy: torch.Tensor


@dataclass
class PolicyStepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    START_NODE: torch.Tensor
    selected_count: int = 0
    current_node: Optional[torch.Tensor] = None
    ninf_mask: Optional[torch.Tensor] = None


class GiantTourEnv:
    """Constraint-blind POMO permutation environment.

    Its action mask contains exactly two rules: depot is permanently masked,
    and already visited customers are masked.  Constraint feasibility is not
    consulted until a complete permutation is passed to exact Split.
    """

    def __init__(self, instance: AdaptedInstance, pomo_size: Optional[int] = None):
        self.instance = instance
        self.depot_xy = instance.policy_view.depot_xy
        self.node_xy = instance.policy_view.node_xy
        self.batch_size, self.problem_size, _ = self.node_xy.shape
        requested = self.problem_size if pomo_size is None else int(pomo_size)
        if requested <= 0 or requested > self.problem_size:
            raise ValueError("pomo_size must be in 1..problem_size")
        self.pomo_size = requested
        self.device = self.node_xy.device

        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(
            self.batch_size, self.pomo_size
        )
        # Deliberately independent of demand/backhaul labels.
        self.START_NODE = torch.arange(1, self.pomo_size + 1, device=self.device)[None].expand(
            self.batch_size, -1
        )
        self.reset_state = PolicyResetState(self.depot_xy, self.node_xy)
        self.step_state = PolicyStepState(self.BATCH_IDX, self.POMO_IDX, self.START_NODE)
        self.last_split_result: Optional[SplitResult] = None
        self.selected_node_list = None

    @classmethod
    def from_official_env(cls, env, pomo_size: Optional[int] = None) -> "GiantTourEnv":
        return cls(MVMoEInstanceAdapter.from_official_env(env), pomo_size=pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            self.batch_size, self.pomo_size, 0, dtype=torch.long, device=self.device
        )
        self.ninf_mask = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1,
            dtype=self.node_xy.dtype, device=self.device,
        )
        self.ninf_mask[:, :, 0] = float("-inf")  # depot is never an action
        self.last_split_result = None
        self._sync_state()
        return self.reset_state, None, False

    def _sync_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

    def pre_step(self):
        self._sync_state()
        return self.step_state, None, False

    def step(self, selected: torch.Tensor):
        if selected.shape != (self.batch_size, self.pomo_size):
            raise ValueError("selected must have shape (batch, pomo)")
        if (selected == 0).any():
            raise ValueError("the depot is permanently masked in a giant-tour policy")
        chosen_mask = self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected]
        if torch.isneginf(chosen_mask).any():
            raise ValueError("selected a depot or an already visited customer")

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")
        self._sync_state()

        done = self.selected_count == self.problem_size
        if not done:
            return self.step_state, None, False

        with torch.no_grad():
            self.last_split_result = split_giant_tours(
                self.depot_xy,
                self.node_xy,
                self.selected_node_list,
                self.instance.split_view,
                return_predecessors=True,
            )
        reward = -self.last_split_result.costs
        return self.step_state, reward, True

    def get_routes(self, batch_index: int, pomo_index: int):
        if self.last_split_result is None or self.last_split_result.predecessors is None:
            raise RuntimeError("finish a rollout before reconstructing routes")
        if not bool(self.last_split_result.feasible[batch_index, pomo_index]):
            raise ValueError("the selected giant tour has no feasible contiguous partition")
        return reconstruct_routes(
            self.selected_node_list[batch_index, pomo_index],
            self.last_split_result.predecessors[batch_index, pomo_index],
        )
