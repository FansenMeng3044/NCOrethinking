from typing import NamedTuple

import torch

from CVRPTWCore import VRPTW_EPSILON
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRPTW(NamedTuple):
    # Fixed tensors are kept once and indexed through ids during shrinking/beam search.
    coords: torch.Tensor
    demand: torch.Tensor
    service_time: torch.Tensor
    tw_start: torch.Tensor
    tw_end: torch.Tensor
    depot_start: torch.Tensor
    depot_end: torch.Tensor
    speed: torch.Tensor
    ids: torch.Tensor

    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    current_time: torch.Tensor
    i: torch.Tensor

    VEHICLE_CAPACITY = 1.0
    EPSILON = VRPTW_EPSILON

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        return mask_long2bool(self.visited_, n=self.demand.size(-1))

    def __getitem__(self, key):
        if not (torch.is_tensor(key) or isinstance(key, slice)):
            raise TypeError("state indices must be tensors or slices")
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            current_time=self.current_time[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        depot, loc, demand = input["depot"], input["loc"], input["demand"]
        batch_size, n_loc, _ = loc.shape
        device, dtype = loc.device, loc.dtype

        def batch_scalar(name, default):
            value = input.get(name)
            if value is None:
                return torch.full((batch_size, 1), default, device=device, dtype=dtype)
            return value.to(device=device, dtype=dtype).reshape(batch_size, 1)

        depot_start = batch_scalar("depot_start", 0.0)
        return StateCVRPTW(
            coords=torch.cat((depot[:, None], loc), dim=1),
            demand=demand,
            service_time=input["service_time"],
            tw_start=input["tw_start"],
            tw_end=input["tw_end"],
            depot_start=depot_start,
            depot_end=batch_scalar("depot_end", 3.0),
            speed=batch_scalar("speed", 1.0),
            ids=torch.arange(batch_size, device=device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(
                torch.zeros(batch_size, 1, n_loc + 1, dtype=torch.uint8, device=device)
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size, 1, (n_loc + 63) // 64,
                    dtype=torch.int64, device=device,
                )
            ),
            lengths=demand.new_zeros(batch_size, 1),
            cur_coord=depot[:, None],
            current_time=depot_start.clone(),
            i=torch.zeros(1, dtype=torch.long, device=device),
        )

    def update(self, selected):
        if self.i.size(0) != 1:
            raise RuntimeError("StateCVRPTW.update supports a single decoding step")
        selected = selected[:, None]
        n_loc = self.demand.size(-1)
        current_coord = self.coords[self.ids, selected]
        travel = (current_coord - self.cur_coord).norm(p=2, dim=-1)
        lengths = self.lengths + travel
        customer_index = (selected - 1).clamp(min=0, max=n_loc - 1)
        selected_demand = self.demand[self.ids, customer_index]
        used_capacity = (self.used_capacity + selected_demand) * (selected != 0).float()

        tw_start_with_depot = torch.cat((self.depot_start, self.tw_start), dim=1)
        service_with_depot = torch.cat(
            (self.service_time.new_zeros(self.service_time.size(0), 1), self.service_time),
            dim=1,
        )
        selected_start = tw_start_with_depot[self.ids, selected]
        selected_service = service_with_depot[self.ids, selected]
        completion = torch.maximum(
            self.current_time + travel / self.speed[self.ids, 0], selected_start
        ) + selected_service
        current_time = torch.where(
            selected == 0, self.depot_start[self.ids, 0], completion
        )

        if self.visited_.dtype == torch.uint8:
            visited = self.visited_.scatter(-1, selected[:, :, None], 1)
        else:
            visited = mask_long_scatter(self.visited_, selected - 1)
        return self._replace(
            prev_a=selected,
            used_capacity=used_capacity,
            visited_=visited,
            lengths=lengths,
            cur_coord=current_coord,
            current_time=current_time,
            i=self.i + 1,
        )

    def all_finished(self):
        return bool(self.get_finished().all())

    def get_finished(self):
        if self.visited_.dtype == torch.uint8:
            visited_customers = self.visited_[:, :, 1:]
        else:
            visited_customers = mask_long2bool(
                self.visited_, n=self.demand.size(-1)
            )
        return visited_customers.bool().all(dim=-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        if self.visited_.dtype == torch.uint8:
            visited_customers = self.visited_[:, :, 1:].bool()
        else:
            visited_customers = mask_long2bool(
                self.visited_, n=self.demand.size(-1)
            ).bool()
        demand = self.demand[self.ids, :]
        exceeds_capacity = (
            demand + self.used_capacity[:, :, None]
            > self.VEHICLE_CAPACITY + self.EPSILON
        )

        coords = self.coords[self.ids, :, :]
        travel = (coords - self.cur_coord[:, :, None, :]).norm(p=2, dim=-1)
        depot_start = self.depot_start[self.ids, :]
        depot_end = self.depot_end[self.ids, :]
        speed = self.speed[self.ids, :]
        candidate_tw_start = torch.cat((depot_start, self.tw_start[self.ids, :]), dim=2)
        candidate_tw_end = torch.cat((depot_end, self.tw_end[self.ids, :]), dim=2)
        candidate_service = torch.cat(
            (
                self.service_time.new_zeros(self.ids.size(0), 1, 1),
                self.service_time[self.ids, :],
            ),
            dim=2,
        )
        service_start = torch.maximum(
            self.current_time[:, :, None] + travel / speed,
            candidate_tw_start,
        )
        out_of_window = service_start > candidate_tw_end + self.EPSILON
        return_distance = (coords - coords[:, :, :1, :]).norm(p=2, dim=-1)
        cannot_return = (
            service_start + candidate_service + return_distance / speed
            > depot_end + self.EPSILON
        )
        mask_customers = (
            visited_customers | exceeds_capacity | out_of_window[:, :, 1:]
            | cannot_return[:, :, 1:]
        )
        # A depot action closes this vehicle route.  Consecutive depot actions
        # are forbidden while at least one feasible unserved customer remains.
        mask_depot = out_of_window[:, :, 0] | cannot_return[:, :, 0]
        mask_depot |= (self.prev_a == 0) & (~mask_customers).any(dim=-1)
        return torch.cat((mask_depot[:, :, None], mask_customers), dim=-1)

    def get_final_cost(self):
        if not self.all_finished():
            raise RuntimeError("cannot compute final cost before all customers are visited")
        depot = self.coords[self.ids, 0, :]
        return self.lengths + (depot - self.cur_coord).norm(p=2, dim=-1)

    def construct_solutions(self, actions):
        return actions
