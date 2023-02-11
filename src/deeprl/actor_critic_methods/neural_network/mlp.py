from functools import reduce
from typing import Callable, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from attrs import define
from torch import Tensor
from torch.distributions import Distribution, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.types import Number


@define(eq=False, slots=False)
class GaussianPolicy(nn.Module):
    """
    TODO
    Action scaling/unscaling

    TODO
    Why clip log_stdev can improve numerical stability?

    FIXME
    nn.ModuleList[nn.Linear] raises: "ModuleList" expects no type arguments
    https://github.com/pytorch/pytorch/pull/89135
    """

    _lyrs: nn.ModuleList
    _mean_lyr: nn.Linear
    _log_stdev_lyr: nn.Linear
    _actv_fn: Callable[[Tensor], Tensor]

    # Allow customisation for easier testing, and not intended to be passed
    _log_stdev_min: Number = -20
    _log_stdev_max: Number = 2

    def forward(self, state: Tensor) -> Union[Distribution, Tensor]:
        actv = state
        for lyr in self._lyrs:
            actv = self._actv_fn(lyr(actv))
        mean: Tensor = self._mean_lyr(actv)

        if self.training:
            log_stdev: Tensor = self._log_stdev_lyr(actv)
            log_stdev = torch.clamp(log_stdev, self._log_stdev_min, self._log_stdev_max)
            stdev = log_stdev.exp()
            tanh_transform = TanhTransform(cache_size=1)
            return TransformedDistribution(Normal(mean, stdev), tanh_transform)
        else:
            return torch.tanh(mean)

    @classmethod
    def init(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        actv_fn: Callable[[Tensor], Tensor] = F.relu,
    ) -> "GaussianPolicy":
        lyrs = nn.ModuleList([nn.Linear(state_dim, hidden_dims[0])])
        lyrs.extend(
            [
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:])
            ]
        )

        mean_lyr = nn.Linear(hidden_dims[-1], action_dim)
        log_stdev_lyr = nn.Linear(hidden_dims[-1], action_dim)

        return cls(
            lyrs,
            mean_lyr,
            log_stdev_lyr,
            actv_fn,
        )

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        self.apply(_init_weights)


@define(eq=False, slots=False)
# eq=False prevents overriding the default hash method
# slots=False disables slotted class to allow inheritance
class Policy(nn.Module):

    _lyrs: nn.ModuleList
    _actv_fn: Callable[[Tensor], Tensor]
    _out_fn: Callable[[Tensor], Tensor]

    def forward(self, state: Tensor) -> Tensor:
        return self._out_fn(self._lyrs[-1](
            reduce(lambda actv, lyr: self._actv_fn(lyr(actv)), self._lyrs[:-1], state)))  # fmt: skip

    @classmethod
    def init(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        actv_fn: Callable[[Tensor], Tensor] = F.relu,
        out_fn: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> "Policy":
        lyrs = nn.ModuleList([nn.Linear(state_dim, hidden_dims[0])])
        lyrs.extend(
            [
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:])
            ]
        )
        lyrs.append(nn.Linear(hidden_dims[-1], action_dim))

        return cls(
            lyrs,
            actv_fn,
            out_fn,
        )

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        self.apply(_init_weights)


@define(eq=False, slots=False)
class Quality(nn.Module):

    _lyrs: nn.ModuleList
    _actv_fn: Callable[[Tensor], Tensor]

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        actv = torch.cat([state, action], dim=1)
        for lyr in self._lyrs[:-1]:
            actv = self._actv_fn(lyr(actv))
        return self._lyrs[-1](actv)  # returns action quality

    @classmethod
    def init(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        actv_fn: Callable[[Tensor], Tensor] = F.relu,
    ) -> "Quality":
        lyrs = nn.ModuleList([nn.Linear(state_dim + action_dim, hidden_dims[0])])
        lyrs.extend(
            [
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:])
            ]
        )
        lyrs.append(nn.Linear(hidden_dims[-1], 1))

        return cls(
            lyrs,
            actv_fn,
        )

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        self.apply(_init_weights)


@torch.no_grad()
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
