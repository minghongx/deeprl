# from collections.abc import Iterable
from typing import Iterable, Tuple, List  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
from typing import TypeVar

import torch
import torch.nn as nn
from torch import Tensor


class Actor(nn.Module):

    def __init__(self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Iterable[int],
            hidden_layer_activation_func: str,  # TODO: Python 3.11 StrEnum
            output_layer_activation_func: str
    ) -> None:
        super(Actor, self).__init__()

        self._hidden_layer_activation_func = nn.ModuleDict({
            'relu': nn.ReLU()
        })[hidden_layer_activation_func]
        self._output_layer_activation_func = nn.ModuleDict({  # controls the amplitude of the output
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1)
        })[output_layer_activation_func]

        dimensions = [state_dim] + list(hidden_dims) + [action_dim]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])

        self._layers.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        '''
        https://github.com/pytorch/pytorch/issues/47336
        activation = state
        for hidden_layer in self._layers[:-1]:
            activation = self._hidden_layer_activation_func( hidden_layer(activation) )
        action = self._output_layer_activation_func( self._layers[-1](activation) )

        One-liner:
        action = self._output_layer_activation_func(self.layers[-1](
            reduce(lambda activation, hidden_layer: self._hidden_layer_activation_func(hidden_layer(activation)), self.layers[:-1], state)))
        However, torch.jit.script raises torch.jit.frontend.UnsupportedNodeError: Lambda aren't supported
        '''
        activation = state
        last = len(self._layers)
        for current, layer in enumerate(self._layers, start=1):
            if current != last:
                activation = self._hidden_layer_activation_func( layer(activation) )
            else:
                activation = self._output_layer_activation_func( layer(activation) )
        action = activation

        return action


class Critic(nn.Module):

    Tensors = TypeVar('Tensors', Tuple[Tensor, ...], List[Tensor])

    def __init__(self,
            state_dim: int,
            actions_dim: int,
            hidden_dims: Iterable[int],
            hidden_layer_activation_func: str
    ) -> None:
        super(Critic, self).__init__()

        self._hidden_layer_activation_func = nn.ModuleDict({
            'relu': nn.ReLU()
        })[hidden_layer_activation_func]

        dimensions = [state_dim + actions_dim] + list(hidden_dims) + [1]

        self._layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])

        self._layers.apply(_init_weights)

    def forward(self, state: Tensors, actions: Tensors) -> Tensor:

        activation = torch.cat(state + actions, dim=1)
        for hidden_layer in self._layers[:-1]:
            activation = self._hidden_layer_activation_func( hidden_layer(activation) )
        action_value = self._layers[-1](activation)

        return action_value


@torch.no_grad()
def _init_weights(layer: nn.Module) -> None:
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
