from dataclasses import dataclass

import torch

from split.constraints import ConstraintSpec, flags_from_problem


@dataclass(frozen=True)
class PolicyView:
    """Static coordinate view consumed by the neural encoder."""

    depot_xy: torch.Tensor
    node_xy: torch.Tensor


@dataclass(frozen=True)
class AdaptedInstance:
    policy_view: PolicyView
    split_view: ConstraintSpec


class MVMoEInstanceAdapter:
    """Extract coordinate and constraint views from a loaded official env.

    The adapter reads public state produced by ``env.load_problems``.  It never
    calls or modifies the official environment's decoding transition.  B/L
    fields are read by the giant-tour decoding environment; C/TW fields are
    enforced by the downstream Split stage.
    """

    @staticmethod
    def from_official_env(env) -> AdaptedInstance:
        if getattr(env, "depot_node_xy", None) is None:
            raise RuntimeError("call the official environment's load_problems first")
        problem = str(env.problem).upper()
        open_route, backhaul, has_limit, has_tw = flags_from_problem(problem)

        all_xy = env.depot_node_xy
        depot_xy = all_xy[:, :1]
        node_xy = all_xy[:, 1:]
        demand = env.depot_node_demand[:, 1:]
        batch = node_xy.size(0)
        capacity = torch.ones(batch, device=demand.device, dtype=demand.dtype)

        route_limit = None
        if has_limit:
            route_limit = getattr(env, "route_limit", None)
            if route_limit is None:
                raise RuntimeError(f"official {problem} env did not expose route_limit")
            route_limit = route_limit.reshape(batch)

        service_time = tw_start = tw_end = depot_start = depot_end = speed = None
        if has_tw:
            service_time = env.depot_node_service_time[:, 1:]
            tw_start = env.depot_node_tw_start[:, 1:]
            tw_end = env.depot_node_tw_end[:, 1:]
            depot_start = env.depot_node_tw_start[:, 0]
            depot_end = env.depot_node_tw_end[:, 0]
            speed = torch.full(
                (batch,), float(env.speed), device=node_xy.device, dtype=node_xy.dtype
            )

        spec = ConstraintSpec(
            problem=problem,
            demand=demand,
            capacity=capacity,
            open_route=open_route,
            backhaul=backhaul,
            has_route_limit=has_limit,
            has_time_windows=has_tw,
            route_limit=route_limit,
            service_time=service_time,
            tw_start=tw_start,
            tw_end=tw_end,
            depot_start=depot_start,
            depot_end=depot_end,
            speed=speed,
            loc_scaler=getattr(env, "loc_scaler", None),
        )
        spec.validate(node_xy)
        return AdaptedInstance(PolicyView(depot_xy=depot_xy, node_xy=node_xy), spec)
