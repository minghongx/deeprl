# from collections.abc import Iterable
# TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        activation_func: str,  # TODO: Python 3.11 StrEnum
        output_func: str,  # controls the amplitude of the output
    ) -> None:
        super(Actor, self).__init__()

        # fmt: off
        self._activation_func = nn.ModuleDict({
            'relu': nn.ReLU(),
        })[activation_func]
        self._output_func = nn.ModuleDict({
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1),
        })[output_func]
        # fmt: on

        dimensions = [state_dim] + list(hidden_dims) + [action_dim]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])  # fmt: skip

        self._layers.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        # https://github.com/pytorch/pytorch/issues/47336
        # activation = state
        # for hidden_layer in self._layers[:-1]:
        #     activation = self._activation_func( hidden_layer(activation) )
        # action = self._output_func( self._layers[-1](activation) )
        # One-liner:
        # action = self._output_func(self.layers[-1](
        #     reduce(lambda activation, hidden_layer: self._activation_func(hidden_layer(activation)), self.layers[:-1], state)))
        # However, torch.jit.script raises torch.jit.frontend.UnsupportedNodeError: Lambda aren't supported
        activation = state
        last = len(self._layers)
        for current, layer in enumerate(self._layers, start=1):
            if current != last:
                activation = self._activation_func(layer(activation))
            else:
                activation = self._output_func(layer(activation))
        action = activation

        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        activation_func: str,
    ) -> None:
        super(Critic, self).__init__()

        # fmt: off
        self.activation_func = nn.ModuleDict({
            'relu': nn.ReLU(),
        })[activation_func]
        # fmt: on

        dimensions = [state_dim + action_dim] + list(hidden_dims) + [1]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])  # fmt: skip

        self._layers.apply(_init_weights)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        activation = torch.cat([state, action], dim=1)
        for hidden_layer in self._layers[:-1]:
            activation = self.activation_func(hidden_layer(activation))
        action_value = self._layers[-1](activation)

        return action_value


@torch.no_grad()
def _init_weights(layer: nn.Module) -> None:
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
