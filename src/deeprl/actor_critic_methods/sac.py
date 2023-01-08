import math
from copy import deepcopy
from functools import partial

# from collections.abc import Callable, Iterator
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Callable,
    Iterable,
    Iterator,
)

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
        device: torch.device,
        state_dim: int,
        action_dim: int,
        policy: Callable[[int, int], StochasticActor],
        critic: Callable[[int, int], ActionCritic],
        policy_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        temperature_optimiser: Callable[[Iterable[Tensor]], Optimizer],
        experience_replay: ExperienceReplay,
        batch_size: int,
        discount_factor: float,
        target_smoothing_factor: float,  # Exponential smoothing
        num_critics: int = 2,
    ) -> None:

        self._policy = policy(state_dim, action_dim).to(device)
        self._critics = [
            deepcopy(critic(state_dim, action_dim).to(device))
            for _ in range(num_critics)
        ]
        self._target_critics = deepcopy(self._critics)
        # Freeze target critics with respect to optimisers (only update via Polyak averaging)
        [net.requires_grad_(False) for net in self._target_critics]

        self._policy_optimiser = policy_optimiser(self._policy.parameters())
        self._critic_optimisers = [
            critic_optimiser(critic.parameters()) for critic in self._critics
        ]

        self._experience_replay = experience_replay
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._target_smoothing_factor = target_smoothing_factor

        # Using log value of temperature in temperature loss are generally nicer TODO: Why?
        # https://github.com/toshikwa/soft-actor-critic.pytorch/issues/2
        self._log_temperature = torch.zeros(1, requires_grad=True, device=device)
        self._temperature_optimiser = temperature_optimiser([self._log_temperature])

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
        # fmt: off

        # Abbreviating to mathematical italic unicode char for readability
        𝑠 = batch.states
        𝘢 = batch.actions
        𝑟 = batch.rewards
        𝑠ʼ = batch.next_states
        𝑑 = batch.terminateds
        𝛾 = self._discount_factor
        𝑄_ = self._critics
        𝑄ʼ_ = self._target_critics
        𝜏 = self._target_smoothing_factor
        log𝛼 = self._log_temperature
        𝛼 = logα.exp().detach()  # FIXME
        𝓗 = self._target_entropy
        """
        𝜇 denotes the action distribution with infinite support (unbounded Gaussian)
        𝜋 denotes the tanh squashed 𝜇
        """

        # Compute target action and its log-likelihood
        𝜇ʼ: Distribution = self._policy(𝑠ʼ)
        uʼ = 𝜇ʼ.rsample()  # Reparameterised sample
        # 𝐄𝐧𝐟𝐨𝐫𝐜𝐢𝐧𝐠 𝐀𝐜𝐭𝐢𝐨𝐧 𝐁𝐨𝐮𝐧𝐝𝐬
        𝘢ʼ = torch.tanh(uʼ)  # Apply an invertible squashing function (tanh) to the Gaussian sample to get bounded action
        log𝜇ʼ = 𝜇ʼ.log_prob(uʼ)
        log𝜋ʼ: Tensor = log𝜇ʼ - 2 * (math.log(2) - uʼ - F.softplus(-2 * uʼ))  # Employ change of variables formula (SAC 2018, app C, eq 21) to compute the likelihood of the bounded action
        """
        The second term is mathematically equivalent to log(1 - tanh(x)^2) but more
        numerically-stable.
        Derivation:
        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))
        """
        log𝜋ʼ = log𝜋ʼ.sum(dim=1, keepdim=True)  # TODO: Why?

        𝑦 = 𝑟 + ~𝑑 * 𝛾 * (min(*[𝑄ʼ(𝑠ʼ, 𝘢ʼ) for 𝑄ʼ in 𝑄ʼ_]) - 𝛼 * logπʼ)  # computes learning target
        action_values = [𝑄(𝑠, 𝘢) for 𝑄 in 𝑄_]
        critic_loss_fn = comp(reduce(add), map(partial(F.mse_loss, target=𝑦)))
        critic_loss: Tensor = critic_loss_fn(action_values)
        [critic_optimiser.zero_grad() for critic_optimiser in self._critic_optimisers]  # type: ignore
        critic_loss.backward()
        [critic_optimiser.step() for critic_optimiser in self._critic_optimisers]

        # Compute action and its log-likelihood
        𝜇: Distribution = self._policy(𝑠)
        u = 𝜇.rsample()
        # 𝐄𝐧𝐟𝐨𝐫𝐜𝐢𝐧𝐠 𝐀𝐜𝐭𝐢𝐨𝐧 𝐁𝐨𝐮𝐧𝐝𝐬
        ã = torch.tanh(u)  # denotes the action sampled fresh from the policy (whereas 𝘢 denotes the action comes from the experience replay)
        log𝜇 = 𝜇.log_prob(u)
        log𝜋: Tensor = log𝜇 - 2 * (math.log(2) - u - F.softplus(-2 * u))
        log𝜋 = log𝜋.sum(dim=1, keepdim=True)
        # fmt: on

        policy_loss = (𝛼 * logπ - min(*[𝑄(𝑠, ã) for 𝑄 in 𝑄_])).mean()
        self._policy_optimiser.zero_grad()
        policy_loss.backward()
        self._policy_optimiser.step()

        temperature_loss = (-log𝛼 * (log𝜋.detach() + 𝓗)).mean()
        self._temperature_optimiser.zero_grad()
        temperature_loss.backward()
        self._temperature_optimiser.step()

        # Update frozen target critics by Polyak averaging (exponential smoothing)
        with torch.no_grad():
            for 𝑄, 𝑄ʼ in zip(𝑄_, 𝑄ʼ_):
                for 𝜃, 𝜃ʼ in zip(𝑄.parameters(), 𝑄ʼ.parameters()):
                    𝜃ʼ.mul_(1.0 - 𝜏)
                    𝜃ʼ.add_(𝜏 * 𝜃)

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        return torch.tanh(self._policy(state).rsample())
