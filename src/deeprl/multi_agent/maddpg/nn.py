# from collections.abc import Iterable
from typing import TypeVar  # fmt: skip
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Iterable,
    List,
    Tuple,
)

import torch
import torch.nn as nn
from torch import Tensor


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        activation_fn: str,  # TODO: Python 3.11 StrEnum
        output_fn: str,
    ) -> None:
        super(Actor, self).__init__()

        # fmt: off
        self._activation_fn = nn.ModuleDict({
            'relu': nn.ReLU(),
        })[activation_fn]
        self._output_fn = nn.ModuleDict({  # controls the amplitude of the output
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1),
        })[output_fn]
        # fmt: on

        dimensions = [state_dim] + list(hidden_dims) + [action_dim]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])  # fmt: skip

        self._layers.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        activation = state
        for hidden_layer in self._layers[:-1]:
            activation = self._activation_fn(hidden_layer(activation))
        return self._output_fn(self._layers[-1](activation))


class Critic(nn.Module):
    Tensors = TypeVar("Tensors", Tuple[Tensor, ...], List[Tensor])

    def __init__(
        self,
        state_dim: int,
        actions_dim: int,
        hidden_dims: Iterable[int],
        activation_fn: str,
    ) -> None:
        super(Critic, self).__init__()

        # fmt: off
        self._activation_fn = nn.ModuleDict({
            'relu': nn.ReLU(),
        })[activation_fn]
        # fmt: on

        dimensions = [state_dim + actions_dim] + list(hidden_dims) + [1]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])  # fmt: skip

        self._layers.apply(_init_weights)

    def forward(self, state: Tensors, actions: Tensors) -> Tensor:
        activation = torch.cat(state + actions, dim=1)
        for hidden_layer in self._layers[:-1]:
            activation = self._activation_fn(hidden_layer(activation))
        return self._layers[-1](activation)


@torch.no_grad()
def _init_weights(layer: nn.Module) -> None:
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
