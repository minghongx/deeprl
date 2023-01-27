"""
TODO 3.9
Generic Alias Type and PEP 585.

TODO
Proper type hint for functools.partial.
"""

from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Callable, Iterable, Iterator, Optional

import torch
import torch.nn.functional as F
from cytoolz import comp
from cytoolz.curried import map, reduce
from torch import Tensor, add, min
from torch.distributions import Distribution
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .experience_replay import ExperienceReplay
from .neural_network import ActionCritic, StochasticActor


class SAC:
    """Soft Actor-Critic"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_init: Callable[[int, int], StochasticActor],
        quality_init: Callable[[int, int], ActionCritic],
        policy_optimiser_init: Callable[[Iterator[Parameter]], Optimizer],
        quality_optimiser_init: Callable[[Iterator[Parameter]], Optimizer],
        temperature_optimiser_init: Callable[[Iterable[Tensor]], Optimizer],
        experience_replay: ExperienceReplay,
        batch_size: int,
        discount_factor: float,
        target_smoothing_factor: float,  # Exponential smoothing
        num_qualities: int = 2,
        device: Optional[torch.device] = None,
    ) -> None:

        self._policy = policy_init(state_dim, action_dim).to(device)
        self._qualities = [
            quality_init(state_dim, action_dim).to(device) for _ in range(num_qualities)
        ]
        self._target_qualities = deepcopy(self._qualities)
        # Freeze target quality networks with respect to optimisers (only update via Polyak averaging)
        [net.requires_grad_(False) for net in self._target_qualities]

        self._policy_optimiser = policy_optimiser_init(self._policy.parameters())
        self._quality_optimiser = quality_optimiser_init(
            chain(*[quality.parameters() for quality in self._qualities])
        )

        self._experience_replay = experience_replay
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._target_smoothing_factor = target_smoothing_factor

        # Using log value of temperature in temperature loss are generally nicer TODO: Why?
        # https://github.com/toshikwa/soft-actor-critic.pytorch/issues/2
        self._log_temperature = torch.zeros(1, requires_grad=True, device=device)
        self._temperature_optimiser = temperature_optimiser_init(
            [self._log_temperature]
        )

        # Differential entropy can be negative TODO: How to understand?
        # https://en.wikipedia.org/wiki/Entropy_(information_theory)#Differential_entropy
        self._target_entropy = -action_dim

    def step(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        terminated: Tensor,
    ) -> None:
        self._experience_replay.push(state, action, reward, next_state, terminated)
        self._update_parameters()

    def _update_parameters(self) -> None:

        try:
            batch = self._experience_replay.sample(self._batch_size)
        except ValueError:
            return

        # Abbreviating to mathematical italic unicode char for readability
        𝑠 = batch.states
        𝘢 = batch.actions
        𝑟 = batch.rewards
        𝑠ʼ = batch.next_states
        𝑑 = batch.terminateds
        𝛾 = self._discount_factor
        𝑄_ = self._qualities
        𝑄ʼ_ = self._target_qualities
        𝜏 = self._target_smoothing_factor
        log𝛼 = self._log_temperature
        𝛼 = logα.exp().detach()
        𝓗 = self._target_entropy

        # Compute target action and its log-likelihood
        𝜋ʼ: Distribution = self._policy(𝑠ʼ)
        𝘢ʼ = 𝜋ʼ.rsample()  # Reparameterised sample
        log𝜋ʼ: Tensor = 𝜋ʼ.log_prob(𝘢ʼ)
        log𝜋ʼ = log𝜋ʼ.sum(dim=1, keepdim=True)  # Sum log prob of multiple actions

        𝑦 = 𝑟 + ~𝑑 * 𝛾 * (min(*[𝑄ʼ(𝑠ʼ, 𝘢ʼ) for 𝑄ʼ in 𝑄ʼ_]) - 𝛼 * logπʼ)
        action_quality = [𝑄(𝑠, 𝘢) for 𝑄 in 𝑄_]
        quality_loss_fn = comp(reduce(add), map(partial(F.mse_loss, target=𝑦)))
        quality_loss: Tensor = quality_loss_fn(action_quality)
        self._quality_optimiser.zero_grad()
        quality_loss.backward()
        self._quality_optimiser.step()

        # Compute action and its log-likelihood
        𝜋: Distribution = self._policy(𝑠)
        ã = 𝜋.rsample()
        log𝜋: Tensor = 𝜋.log_prob(ã)
        log𝜋 = log𝜋.sum(dim=1, keepdim=True)

        policy_loss = (𝛼 * logπ - min(*[𝑄(𝑠, ã) for 𝑄 in 𝑄_])).mean()
        self._policy_optimiser.zero_grad()
        policy_loss.backward()
        self._policy_optimiser.step()

        temperature_loss = (-log𝛼 * (log𝜋.detach() + 𝓗)).mean()
        self._temperature_optimiser.zero_grad()
        temperature_loss.backward()
        self._temperature_optimiser.step()

        # Update frozen target quality fn approximators by Polyak averaging (exponential smoothing)
        with torch.no_grad():
            for 𝑄, 𝑄ʼ in zip(𝑄_, 𝑄ʼ_):
                for 𝜃, 𝜃ʼ in zip(𝑄.parameters(), 𝑄ʼ.parameters()):
                    𝜃ʼ.copy_(𝜏 * 𝜃 + (1.0 - 𝜏) * 𝜃ʼ)

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        return self._policy(state).rsample()
