from typing import NamedTuple

import torch


class StateAMSplit(NamedTuple):
    """Direct-style decoder state for a depot-masked customer permutation."""

    coords: torch.Tensor
    demand: torch.Tensor
    service_time: torch.Tensor
    tw_start: torch.Tensor
    depot_start: torch.Tensor
    speed: torch.Tensor
    ids: torch.Tensor
    first_a: torch.Tensor
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    cur_coord: torch.Tensor
    current_time: torch.Tensor
    visited_: torch.Tensor
    i: torch.Tensor

    @property
    def visited(self):
        return self.visited_ > 0

    def __getitem__(self, key):
        if not (torch.is_tensor(key) or isinstance(key, slice)):
            raise TypeError("state indices must be tensors or slices")
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            cur_coord=self.cur_coord[key],
            current_time=self.current_time[key],
            visited_=self.visited_[key],
        )

    @staticmethod
    def initialize(
        input, visited_dtype=torch.uint8,
        depot_start_default=0.0, speed_default=1.0,
    ):
        if visited_dtype != torch.uint8:
            raise NotImplementedError("compressed masks are not implemented for AM Split")
        depot, loc, demand = input["depot"], input["loc"], input["demand"]
        coords = torch.cat((depot[:, None, :], loc), dim=1)
        batch_size, node_count, _ = coords.size()
        device, dtype = loc.device, loc.dtype

        def batch_scalar(name, default):
            value = input.get(name)
            if value is None:
                return torch.full((batch_size, 1), default, device=device, dtype=dtype)
            return value.to(device=device, dtype=dtype).reshape(batch_size, 1)

        service_time = input.get("service_time")
        if service_time is None:
            service_time = torch.zeros_like(demand)
        tw_start = input.get("tw_start")
        if tw_start is None:
            tw_start = torch.zeros_like(demand)
        depot_start = batch_scalar("depot_start", depot_start_default)
        speed = batch_scalar("speed", speed_default)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=coords.device)
        visited = torch.zeros(
            batch_size, 1, node_count, dtype=torch.uint8, device=coords.device
        )
        # Index 0 is encoded as context but is never a decoder action.
        visited[:, :, 0] = 1
        return StateAMSplit(
            coords=coords,
            demand=demand,
            service_time=service_time,
            tw_start=tw_start,
            depot_start=depot_start,
            speed=speed,
            ids=torch.arange(batch_size, dtype=torch.long, device=coords.device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            used_capacity=demand.new_zeros(batch_size, 1),
            cur_coord=depot[:, None, :],
            current_time=depot_start.clone(),
            visited_=visited,
            i=torch.zeros(1, dtype=torch.long, device=coords.device),
        )

    def update(self, selected):
        prev_a = selected[:, None]
        current_coord = self.coords[self.ids, prev_a]
        travel = torch.linalg.vector_norm(current_coord - self.cur_coord, dim=-1)
        customer_index = (prev_a - 1).clamp(min=0, max=self.demand.size(-1) - 1)
        selected_demand = self.demand[self.ids, customer_index]
        used_capacity = (
            self.used_capacity + selected_demand
        ) * (prev_a != 0).to(self.used_capacity.dtype)

        tw_start_with_depot = torch.cat((self.depot_start, self.tw_start), dim=1)
        service_with_depot = torch.cat(
            (
                self.service_time.new_zeros(self.service_time.size(0), 1),
                self.service_time,
            ),
            dim=1,
        )
        selected_start = tw_start_with_depot[self.ids, prev_a]
        selected_service = service_with_depot[self.ids, prev_a]
        completion = torch.maximum(
            self.current_time + travel / self.speed[self.ids, 0], selected_start
        ) + selected_service
        current_time = torch.where(
            prev_a == 0, self.depot_start[self.ids, 0], completion
        )
        first_a = prev_a if self.i.item() == 0 else self.first_a
        visited = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        return self._replace(
            first_a=first_a,
            prev_a=prev_a,
            used_capacity=used_capacity,
            cur_coord=current_coord,
            current_time=current_time,
            visited_=visited,
            i=self.i + 1,
        )

    def all_finished(self):
        # coords contains one depot plus n customers.
        return self.i.item() >= self.coords.size(1) - 1

    def get_finished(self):
        return torch.full(
            (self.ids.size(0), 1),
            self.all_finished(),
            dtype=torch.bool,
            device=self.ids.device,
        )

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited

    def construct_solutions(self, actions):
        return actions
