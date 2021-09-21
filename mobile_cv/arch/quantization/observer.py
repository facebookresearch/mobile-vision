#!/usr/bin/env python3

from abc import abstractmethod

import torch
from torch.quantization.observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    ObserverBase,
)


def update_stat(m):
    """Wrapper can be called by model.apply to update stats"""
    if isinstance(m, ObserverBase) and hasattr(m, "update_stat") and m.training:
        m.update_stat()


class FixedMinMaxObserver(MinMaxObserver):
    """Hard-coded min-max settings."""

    def __init__(self, *args, fixed_min_val=0, fixed_max_val=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val = torch.tensor(float(fixed_min_val))
        self.max_val = torch.tensor(float(fixed_max_val))

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        return x_orig


class UpdatableMovingAverageMaxStatObserver(MovingAverageMinMaxObserver):
    """
    A moving average observer that has an update_stat method to
    decouple the action of observing tensors from the action of updating the
    stats.

    Max stat means that this observer keeps track of max(abs(input))

    This observer has the additional feature where it minimizes outliers by
    requiring that:
        max_stat <= mean(input) + 4 * std(input)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("max_stat", torch.tensor(float("inf")))

    @abstractmethod
    def update_stat(self) -> None:
        pass

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        max_stat = self.max_stat
        cur_max_stat = torch.max(torch.abs(x))
        # Updates the max stat by truncating at 4 sigma
        cur_max_stat = torch.min(torch.mean(x) + torch.std(x) * 4, cur_max_stat)
        if max_stat == float("inf"):
            max_stat = cur_max_stat
        else:
            max_stat = max_stat + self.averaging_constant * (cur_max_stat - max_stat)
        self.max_stat.copy_(max_stat)
        return x_orig

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Try setting max_stat to max_val if missing"""
        key = prefix + "max_stat"
        backup_key = prefix + "max_val"
        if key not in state_dict and backup_key in state_dict:
            state_dict[key] = state_dict[backup_key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class UpdatableSymmetricMovingAverageMinMaxObserver(
    UpdatableMovingAverageMaxStatObserver
):
    """Assumes activations are symmetric about the zero point"""

    def update_stat(self) -> None:
        if self.max_stat == torch.tensor(float("inf")):
            return

        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch.Tensor.__neg__)[[N...
        self.min_val = -self.max_stat
        # pyre-fixme[8]: Attribute has type `Tensor`; used as `Union[torch.Tensor,
        #  torch.nn.Module]`.
        self.max_val = self.max_stat


class UpdateableReLUMovingAverageMinMaxObserver(UpdatableMovingAverageMaxStatObserver):
    """Assumes activations are non-negative and include the zero point"""

    def update_stat(self) -> None:
        if self.max_stat == torch.tensor(float("inf")):
            return

        self.min_val = torch.tensor(float(0), device=self.max_stat.device)
        # pyre-fixme[8]: Attribute has type `Tensor`; used as `Union[torch.Tensor,
        #  torch.nn.Module]`.
        self.max_val = self.max_stat
