"""
Author's PyTorch implementation of TD3 for OpenAI gym tasks
https://github.com/sfujim/TD3/blob/master/TD3.py
"""

from copy import deepcopy
from functools import partial
from itertools import count

# from collections.abc import Callable, Iterator
from typing import Union  # TODO: Unnecessary since version 3.10. See PEP 604.
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Callable,
    Iterator,
)

import torch
import torch.nn.functional as F
from cytoolz import comp
from cytoolz.curried import map, reduce
from torch import Tensor, add, min
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .experience_replay import ExperienceReplay
from .neural_network.mlp import Actor, Critic
from .noise_injection.action_space import Gaussian


class TD3:
    """Twin-Delayed DDPG"""

    def __init__(
        self,
        policy: Actor,
        critic: Critic,
        policy_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        experience_replay: ExperienceReplay,
        batch_size: int,
        discount_factor: float,
        polyak: float,
        policy_noise: Union[Gaussian, None],
        smoothing_noise_stddev: float,
        smoothing_noise_clip: float,
        num_critics: int = 2,
        policy_delay: int = 2,
    ) -> None:

        self._policy = policy
        self._critics = [deepcopy(critic) for _ in range(num_critics)]
        self._target_policy = deepcopy(self._policy)
        self._target_critics = deepcopy(self._critics)
        # Freeze target networks with respect to optimisers (only update via Polyak averaging)
        self._target_policy.requires_grad_(False)
        [net.requires_grad_(False) for net in self._target_critics]

        self._policy_optimiser = policy_optimiser(self._policy.parameters())
        self._critic_optimisers = [
            critic_optimiser(critic.parameters()) for critic in self._critics
        ]

        self._experience_replay = experience_replay
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._polyak = polyak
        self._policy_noise = policy_noise
        self._smoothing_noise_clip = smoothing_noise_clip
        self._smoothing_noise_stddev = smoothing_noise_stddev
        self._policy_delay = policy_delay

        self._count = count(start=1, step=1)

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
        # fmt: off
        𝑠 = batch.states
        𝘢 = batch.actions
        𝑟 = batch.rewards
        𝑠ʼ = batch.next_states
        𝑑 = batch.terminateds
        𝛾 = self._discount_factor
        𝜎 = self._smoothing_noise_stddev
        𝑐 = self._smoothing_noise_clip
        𝜇 = self._policy  # Deterministic policy is usually denoted by 𝜇, and stochastic " 𝜋
        𝜇ʼ = self._target_policy
        𝑄_ = self._critics
        𝑄ʼ_ = self._target_critics
        𝜌 = self._polyak
        # fmt: on

        # Compute target action
        𝘢ʼ: Tensor = 𝜇ʼ(𝑠ʼ)

        # Target policy smoothing: add clipped noise to the target action
        ã = 𝘢ʼ + 𝘢ʼ.clone().normal_(0, 𝜎).clamp_(-𝑐, 𝑐)
        ã.clamp_(-1, 1)  # clipped to lie in valid action range

        # Clipped double-Q learning
        𝑦 = 𝑟 + ~𝑑 * 𝛾 * min(*[𝑄ʼ(𝑠ʼ, ã) for 𝑄ʼ in 𝑄ʼ_])  # computes learning target
        action_values = [𝑄(𝑠, 𝘢) for 𝑄 in 𝑄_]
        critic_loss_func = comp(reduce(add), map(partial(F.mse_loss, target=𝑦)))
        critic_loss = critic_loss_func(action_values)
        [critic_optimiser.zero_grad() for critic_optimiser in self._critic_optimisers]  # type: ignore
        critic_loss.backward()
        [critic_optimiser.step() for critic_optimiser in self._critic_optimisers]

        # "Delayed" policy updates
        if next(self._count) % self._policy_delay == 0:

            # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
            policy_loss: Tensor = -𝑄_[0](𝑠, 𝜇(𝑠)).mean()
            self._policy_optimiser.zero_grad()
            policy_loss.backward()
            self._policy_optimiser.step()

            # Update frozen target networks by Polyak averaging
            with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
                for 𝑄, 𝑄ʼ in zip(𝑄_, 𝑄ʼ_):
                    for 𝜙, 𝜙ʼ in zip(𝑄.parameters(), 𝑄ʼ.parameters()):
                        𝜙ʼ.mul_(𝜌)
                        𝜙ʼ.add_((1.0 - 𝜌) * 𝜙)
                for 𝜃, 𝜃ʼ in zip(𝜇.parameters(), 𝜇ʼ.parameters()):
                    𝜃ʼ.mul_(𝜌)
                    𝜃ʼ.add_((1.0 - 𝜌) * 𝜃)

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        action: Tensor = self._policy(state)
        # TODO: Avaliable since version 3.10. See PEP 634
        # match self._policy_noise:
        #     case Gaussian():
        #     case _:
        if isinstance(self._policy_noise, Gaussian):
            action += self._policy_noise(action.size(), action.device)
            action.clamp_(-1, 1)  # Output layer of actor network is tanh activated
        return action
