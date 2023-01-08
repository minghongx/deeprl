from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class DeterministicActor(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        ...


class StochasticActor(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor) -> Distribution:
        ...


class ActionCritic(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        ...
