from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch


FEASIBILITY_EPSILON = 1e-5
"""Feasibility tolerance used by the official MVMoE environments/generators."""


ALL_PROBLEMS: Tuple[str, ...] = (
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW",
)


def flags_from_problem(problem: str) -> Tuple[bool, bool, bool, bool]:
    """Return (open, backhaul, route_limit, time_windows) for an official name."""
    name = problem.upper()
    if name not in ALL_PROBLEMS:
        raise ValueError(f"unsupported MVMoE problem: {problem!r}")
    return name.startswith("O"), "B" in name, "L" in name, "TW" in name


@dataclass(frozen=True)
class ConstraintSpec:
    """Official constraint data used across decoder state, Split, and replay.

    MVMoE normalizes demands by per-instance vehicle capacity before exposing
    them to an environment, hence ``capacity`` defaults to one. B is consumed
    during order decoding and then rechecked by Split; L, C, and TW are consumed
    only by Split. Tensor fields have batch as their first dimension. Customer
    fields have shape (B, n).
    """

    problem: str
    demand: torch.Tensor
    capacity: torch.Tensor
    open_route: bool = False
    backhaul: bool = False
    has_route_limit: bool = False
    has_time_windows: bool = False
    route_limit: Optional[torch.Tensor] = None
    service_time: Optional[torch.Tensor] = None
    tw_start: Optional[torch.Tensor] = None
    tw_end: Optional[torch.Tensor] = None
    depot_start: Optional[torch.Tensor] = None
    depot_end: Optional[torch.Tensor] = None
    speed: Optional[torch.Tensor] = None
    loc_scaler: Optional[float] = None

    def validate(self, node_xy: torch.Tensor) -> None:
        if node_xy.ndim != 3 or node_xy.size(-1) != 2:
            raise ValueError("node_xy must have shape (batch, n, 2)")
        batch, n, _ = node_xy.shape
        if self.demand.shape != (batch, n):
            raise ValueError("demand must have shape (batch, n)")
        if self.capacity.shape not in ((batch,), (batch, 1)):
            raise ValueError("capacity must have shape (batch,) or (batch, 1)")
        if not torch.isfinite(self.demand).all():
            raise ValueError("demand contains a non-finite value")
        if not torch.isfinite(self.capacity).all() or (self.capacity <= 0).any():
            raise ValueError("capacity must be finite and positive")
        if not self.backhaul and (self.demand < 0).any():
            raise ValueError("negative demand is only valid for backhaul problems")
        if self.has_route_limit:
            if self.route_limit is None:
                raise ValueError("route_limit is required when has_route_limit=True")
            if self.route_limit.shape not in ((batch,), (batch, 1)):
                raise ValueError("route_limit must have shape (batch,) or (batch, 1)")
            if not torch.isfinite(self.route_limit).all() or (self.route_limit <= 0).any():
                raise ValueError("route_limit must be finite and positive")
        if self.has_time_windows:
            fields = {
                "service_time": self.service_time,
                "tw_start": self.tw_start,
                "tw_end": self.tw_end,
            }
            for name, value in fields.items():
                if value is None or value.shape != (batch, n):
                    raise ValueError(f"{name} must have shape (batch, n)")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains a non-finite value")
            for name, value in {
                "depot_start": self.depot_start,
                "depot_end": self.depot_end,
                "speed": self.speed,
            }.items():
                if value is None or value.shape not in ((batch,), (batch, 1)):
                    raise ValueError(f"{name} must have shape (batch,) or (batch, 1)")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains a non-finite value")
            if (self.tw_start > self.tw_end).any():
                raise ValueError("a customer time-window start exceeds its end")
            if (self.depot_start > self.depot_end).any():
                raise ValueError("a depot start exceeds its end")
            if (self.speed <= 0).any():
                raise ValueError("speed must be positive")
        expected = flags_from_problem(self.problem)
        actual = (self.open_route, self.backhaul, self.has_route_limit, self.has_time_windows)
        if expected != actual:
            raise ValueError(
                f"constraint flags {actual} disagree with official problem {self.problem}={expected}"
            )

    def to(self, device: torch.device) -> "ConstraintSpec":
        values = {}
        for name in (
            "demand", "capacity", "route_limit", "service_time", "tw_start",
            "tw_end", "depot_start", "depot_end", "speed",
        ):
            value = getattr(self, name)
            values[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return replace(self, **values)
