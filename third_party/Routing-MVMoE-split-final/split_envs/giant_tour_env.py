from dataclasses import dataclass
from typing import Optional

import torch

from split.constraints import FEASIBILITY_EPSILON
from split.decoder import SplitResult, reconstruct_routes, split_giant_tours
from .adapter import AdaptedInstance, MVMoEInstanceAdapter


@dataclass
class PolicyResetState:
    # Static node encoding remains strictly coordinate-only.  B enters only
    # through the dynamic decoder state below; L is reserved for Split.
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
    b_context: Optional[torch.Tensor] = None
    b_candidate: Optional[torch.Tensor] = None


class GiantTourEnv:
    """B-aware POMO ordering environment followed by B/L/C/TW Split.

    Node embeddings consume coordinates only. During decoding, backhaul (B)
    signed-load transitions are represented by deterministic hidden route
    starts. Each candidate is tested both as a continuation and as the first
    customer of a new route under the official MVMoE semantics. Route length
    (L) is deliberately absent from decoder features, masks, and transitions.
    The policy emits one permutation and never selects the depot. Final Split
    independently enforces B/L/C/TW and is free to replace the decoder's hidden
    B-feasibility witness with the best feasible contiguous partition.
    """

    _EPSILON = FEASIBILITY_EPSILON

    def __init__(self, instance: AdaptedInstance, pomo_size: Optional[int] = None):
        self.instance = instance
        self.depot_xy = instance.policy_view.depot_xy
        self.node_xy = instance.policy_view.node_xy
        self.batch_size, self.problem_size, _ = self.node_xy.shape
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
        self.reset_state = PolicyResetState(self.depot_xy, self.node_xy)
        self.step_state = PolicyStepState(self.BATCH_IDX, self.POMO_IDX, self.START_NODE)
        self.last_split_result: Optional[SplitResult] = None
        self.selected_node_list = None
        self.mandatory_breaks = None

    @classmethod
    def from_official_env(cls, env, pomo_size: Optional[int] = None) -> "GiantTourEnv":
        return cls(MVMoEInstanceAdapter.from_official_env(env), pomo_size=pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            self.batch_size, self.pomo_size, 0, dtype=torch.long, device=self.device
        )
        self.mandatory_breaks = torch.empty(
            self.batch_size, self.pomo_size, 0, dtype=torch.bool, device=self.device
        )
        self.visited_ninf_mask = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1,
            dtype=self.node_xy.dtype, device=self.device,
        )
        self.visited_ninf_mask[:, :, 0] = float("-inf")
        self.current_backhaul_load = torch.zeros(
            self.batch_size, self.pomo_size, dtype=self.node_xy.dtype, device=self.device
        )
        self.last_split_result = None
        self._update_action_mask()
        self._sync_state()
        return self.reset_state, None, False

    def _sync_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.b_context = self._b_context()
        self.step_state.b_candidate = self.b_candidate

    def _expanded_customer_demand(self):
        return self.instance.split_view.demand[:, None, :].expand(
            -1, self.pomo_size, -1
        )

    def _unvisited_linehauls(self):
        visited = torch.isneginf(self.visited_ninf_mask[:, :, 1:])
        return ((self._expanded_customer_demand() > 0) & ~visited).any(dim=2)

    def _capacity(self):
        spec = self.instance.split_view
        return spec.capacity.reshape(self.batch_size, 1).expand(-1, self.pomo_size)

    def _b_context(self):
        """Return [B enabled, normalized signed load]; never expose L."""
        spec = self.instance.split_view
        enabled_b = torch.full_like(self.current_backhaul_load, float(spec.backhaul))
        if spec.backhaul:
            backhaul_load = self.current_backhaul_load / self._capacity()
        else:
            backhaul_load = torch.zeros_like(self.current_backhaul_load)
        return torch.stack((enabled_b, backhaul_load), dim=2)

    def _candidate_transitions(self):
        """Compute official B continuation and restart transitions only."""
        spec = self.instance.split_view
        demand = self._expanded_customer_demand()
        zeros = torch.zeros_like(demand)
        true = torch.ones_like(demand, dtype=torch.bool)

        if spec.backhaul:
            capacity = self._capacity()
            continue_load = self.current_backhaul_load[:, :, None] - demand
            start_load = torch.where(
                self._unvisited_linehauls(),
                capacity,
                torch.zeros_like(self.current_backhaul_load),
            )
            restart_load = start_load[:, :, None] - demand
            continue_b = (
                (continue_load >= -self._EPSILON)
                & (continue_load <= capacity[:, :, None] + self._EPSILON)
            )
            restart_b = (
                (restart_load >= -self._EPSILON)
                & (restart_load <= capacity[:, :, None] + self._EPSILON)
            )
        else:
            continue_load = restart_load = zeros
            continue_b = restart_b = true

        return continue_b, restart_b, continue_load, restart_load

    def _update_action_mask(self):
        self.ninf_mask = self.visited_ninf_mask.clone()
        transitions = self._candidate_transitions()
        (
            self._can_continue, self._can_restart,
            self._continue_load, self._restart_load,
        ) = transitions
        if self.selected_count == 0:
            admissible = self._can_restart
        else:
            admissible = self._can_continue | self._can_restart
        self.ninf_mask[:, :, 1:] = self.ninf_mask[:, :, 1:].masked_fill(
            ~admissible, float("-inf")
        )

        spec = self.instance.split_view
        self.b_candidate = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1, 3,
            dtype=self.node_xy.dtype, device=self.device,
        )
        if spec.backhaul:
            customer_features = torch.stack(
                (
                    self._expanded_customer_demand() / self._capacity()[:, :, None],
                    self._can_continue.to(self.node_xy.dtype),
                    ((~self._can_continue) & self._can_restart).to(self.node_xy.dtype),
                ),
                dim=3,
            )
            self.b_candidate[:, :, 1:] = customer_features

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
        can_continue = self._can_continue[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        can_restart = self._can_restart[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        route_break = torch.ones_like(can_continue) if self.selected_count == 0 else ~can_continue
        if (route_break & ~can_restart).any():
            raise RuntimeError("selected customer has no feasible B restart transition")
        continue_load = self._continue_load[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        restart_load = self._restart_load[
            self.BATCH_IDX, self.POMO_IDX, customer_index
        ]
        next_load = torch.where(route_break, restart_load, continue_load)

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        self.mandatory_breaks = torch.cat(
            (self.mandatory_breaks, route_break[:, :, None]), dim=2
        )
        self.current_backhaul_load = next_load
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
