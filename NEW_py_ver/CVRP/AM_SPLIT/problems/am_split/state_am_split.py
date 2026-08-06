from typing import NamedTuple

import torch


class StateAMSplit(NamedTuple):
    """Decoder state for a depot-masked permutation of all customers."""

    coords: torch.Tensor
    ids: torch.Tensor
    first_a: torch.Tensor
    prev_a: torch.Tensor
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
            visited_=self.visited_[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        if visited_dtype != torch.uint8:
            raise NotImplementedError("compressed masks are not implemented for AM Split")
        coords = torch.cat((input["depot"][:, None, :], input["loc"]), dim=1)
        batch_size, node_count, _ = coords.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=coords.device)
        visited = torch.zeros(
            batch_size, 1, node_count, dtype=torch.uint8, device=coords.device
        )
        # Index 0 is encoded as context but is never a decoder action.
        visited[:, :, 0] = 1
        return StateAMSplit(
            coords=coords,
            ids=torch.arange(batch_size, dtype=torch.long, device=coords.device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            visited_=visited,
            i=torch.zeros(1, dtype=torch.long, device=coords.device),
        )

    def update(self, selected):
        prev_a = selected[:, None]
        first_a = prev_a if self.i.item() == 0 else self.first_a
        visited = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        return self._replace(
            first_a=first_a,
            prev_a=prev_a,
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
