from dataclasses import dataclass
from typing import Optional

import torch

from split.constraints import FEASIBILITY_EPSILON
from split.decoder import SplitResult, reconstruct_routes, split_giant_tours
from .adapter import AdaptedInstance, MVMoEInstanceAdapter


@dataclass
class PolicyResetState:
    # Match the original MVMoE static input contract.
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor
    node_tw_start: torch.Tensor
    node_tw_end: torch.Tensor


@dataclass
class PolicyStepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    START_NODE: torch.Tensor
    selected_count: int = 0
    current_node: Optional[torch.Tensor] = None
    ninf_mask: Optional[torch.Tensor] = None
    load: Optional[torch.Tensor] = None
    current_time: Optional[torch.Tensor] = None
    length: Optional[torch.Tensor] = None
    open: Optional[torch.Tensor] = None


class GiantTourEnv:
    """Original MVMoE inputs with a B-only ordering-feasibility mask.

    Customer embeddings consume the same ``(x, y, demand, tw_start, tw_end)``
    features as the original models; non-TW environments supply zero TW fields.
    The decoder receives the same four dynamic attributes as the original
    POMO-MTL/MVMoE policies: remaining load, current time, current route length,
    and the open-route flag. Because a giant tour has no depot actions, these
    quantities evolve continuously along the customer permutation and are not
    reset during policy decoding. The action mask excludes the depot, visited
    customers, and B transitions that cannot continue or begin a feasible
    hidden route. C, TW, O, and L never exclude an action. Final Split
    independently enforces all applicable constraints and may choose different
    route boundaries from the hidden B-feasibility witness.
    """

    _EPSILON = FEASIBILITY_EPSILON

    def __init__(self, instance: AdaptedInstance, pomo_size: Optional[int] = None):
        self.instance = instance
        self.depot_xy = instance.policy_view.depot_xy
        self.node_xy = instance.policy_view.node_xy
        self.node_demand = instance.policy_view.node_demand
        self.node_tw_start = instance.policy_view.node_tw_start
        self.node_tw_end = instance.policy_view.node_tw_end
        self.batch_size, self.problem_size, _ = self.node_xy.shape
        expected = (self.batch_size, self.problem_size)
        for name in ("node_demand", "node_tw_start", "node_tw_end"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        requested = self.problem_size if pomo_size is None else int(pomo_size)
        if requested <= 0 or requested > self.problem_size:
            raise ValueError("pomo_size must be in 1..problem_size")
        spec = self.instance.split_view
        if spec.backhaul:
            # Match the official MVMoE convention: distinct POMO starts are
            # linehauls whenever an instance contains linehauls.  Backhaul-only
            # instances may start from any customer.
            linehaul_count = (spec.demand > 0).sum(dim=1)
            available = torch.where(
                linehaul_count > 0,
                linehaul_count,
                torch.full_like(linehaul_count, self.problem_size),
            )
            requested = min(requested, int(available.min().item()))
        self.pomo_size = requested
        self.device = self.node_xy.device

        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(
            self.batch_size, self.pomo_size
        )
        customer_ids = torch.arange(1, self.problem_size + 1, device=self.device)
        if spec.backhaul:
            demand = spec.demand
            has_linehaul = (demand > 0).any(dim=1, keepdim=True)
            preferred = torch.where(
                has_linehaul,
                demand > 0,
                torch.ones_like(demand, dtype=torch.bool),
            )
            ranks = torch.where(
                preferred,
                customer_ids[None],
                customer_ids[None] + self.problem_size,
            )
            self.START_NODE = ranks.argsort(dim=1)[:, :self.pomo_size] + 1
        else:
            self.START_NODE = customer_ids[:self.pomo_size][None].expand(self.batch_size, -1)
        self.reset_state = PolicyResetState(
            self.depot_xy,
            self.node_xy,
            self.node_demand,
            self.node_tw_start,
            self.node_tw_end,
        )
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
        self.visited_ninf_mask = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1,
            dtype=self.node_xy.dtype, device=self.device,
        )
        self.visited_ninf_mask[:, :, 0] = float("-inf")
        shape = (self.batch_size, self.pomo_size)
        # This hidden state certifies that the prefix admits a B-feasible
        # contiguous partition. It is not an additional decoder feature.
        self._b_route_load = torch.zeros(
            shape, dtype=self.node_demand.dtype, device=self.device
        )
        self.load = torch.ones(
            shape, dtype=self.node_demand.dtype, device=self.device
        )
        self.current_time = torch.zeros(
            shape, dtype=self.node_xy.dtype, device=self.device
        )
        self.length = torch.zeros(
            shape, dtype=self.node_xy.dtype, device=self.device
        )
        self.open = torch.full(
            shape, float(self.instance.split_view.open_route),
            dtype=self.node_xy.dtype, device=self.device,
        )
        self.current_coord = self.depot_xy.expand(-1, self.pomo_size, -1).clone()
        self.last_split_result = None
        self._update_action_mask()
        self._sync_state()
        return self.reset_state, None, False

    def _sync_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.open = self.open

    def _expanded_customer_demand(self):
        return self.instance.split_view.demand[:, None, :].expand(
            -1, self.pomo_size, -1
        )

    def _unvisited_linehauls(self):
        visited = torch.isneginf(self.visited_ninf_mask[:, :, 1:])
        return ((self._expanded_customer_demand() > 0) & ~visited).any(dim=2)

    def _capacity(self):
        return self.instance.split_view.capacity.reshape(
            self.batch_size, 1
        ).expand(-1, self.pomo_size)

    def _b_candidate_transitions(self):
        """Return B-feasible continuation/restart transitions for every customer."""
        spec = self.instance.split_view
        demand = self._expanded_customer_demand()
        zeros = torch.zeros_like(demand)
        admissible = torch.ones_like(demand, dtype=torch.bool)
        if not spec.backhaul:
            return admissible, admissible, zeros, zeros

        capacity = self._capacity()
        continue_load = self._b_route_load[:, :, None] - demand
        start_load = torch.where(
            self._unvisited_linehauls(),
            capacity,
            torch.zeros_like(self._b_route_load),
        )
        restart_load = start_load[:, :, None] - demand
        can_continue = (
            (continue_load >= -self._EPSILON)
            & (continue_load <= capacity[:, :, None] + self._EPSILON)
        )
        can_restart = (
            (restart_load >= -self._EPSILON)
            & (restart_load <= capacity[:, :, None] + self._EPSILON)
        )
        return can_continue, can_restart, continue_load, restart_load

    def _update_action_mask(self):
        self.ninf_mask = self.visited_ninf_mask.clone()
        (
            self._b_can_continue,
            self._b_can_restart,
            self._b_continue_load,
            self._b_restart_load,
        ) = self._b_candidate_transitions()
        if self.selected_count == 0:
            admissible = self._b_can_restart
        else:
            admissible = self._b_can_continue | self._b_can_restart
        self.ninf_mask[:, :, 1:] = self.ninf_mask[:, :, 1:].masked_fill(
            ~admissible, float("-inf")
        )

    def pre_step(self):
        self._sync_state()
        return self.step_state, None, False

    def step(self, selected: torch.Tensor):
        if selected.shape != (self.batch_size, self.pomo_size):
            raise ValueError("selected must have shape (batch, pomo)")
        if (selected == 0).any():
            raise ValueError("the depot is not an action in a giant-tour policy")
        chosen_mask = self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected]
        if torch.isneginf(chosen_mask).any():
            raise ValueError("selected a customer forbidden by visit or B constraints")

        customer_index = selected - 1
        spec = self.instance.split_view
        can_continue = self._b_can_continue[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        can_restart = self._b_can_restart[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        hidden_break = (
            torch.ones_like(can_continue)
            if self.selected_count == 0
            else ~can_continue
        )
        if (hidden_break & ~can_restart).any():
            raise RuntimeError("selected customer has no feasible B restart transition")
        continue_load = self._b_continue_load[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        restart_load = self._b_restart_load[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        next_b_load = torch.where(hidden_break, restart_load, continue_load)
        selected_coord = self.node_xy[self.BATCH_IDX, customer_index]
        selected_demand = spec.demand[self.BATCH_IDX, customer_index]
        travel = torch.linalg.vector_norm(selected_coord - self.current_coord, dim=2)

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        self.load = self.load - selected_demand
        self.length = self.length + travel
        if spec.has_time_windows:
            speed = spec.speed[:, None].expand(-1, self.pomo_size)
            selected_tw_start = spec.tw_start[self.BATCH_IDX, customer_index]
            selected_service = spec.service_time[self.BATCH_IDX, customer_index]
            service_start = torch.maximum(
                self.current_time + travel / speed, selected_tw_start
            )
            self.current_time = service_start + selected_service
        self.current_coord = selected_coord
        self._b_route_load = next_b_load
        self.visited_ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")
        self._update_action_mask()
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
