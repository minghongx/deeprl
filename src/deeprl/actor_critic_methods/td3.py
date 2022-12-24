"""
Author's PyTorch implementation of TD3 for OpenAI gym tasks
https://github.com/sfujim/TD3/blob/master/TD3.py
"""

from copy import deepcopy
from itertools import count

# from collections.abc import Callable, Iterator
from typing import Union  # TODO: Unnecessary since version 3.10. See PEP 604.
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Callable,
    Iterator,
)

import torch
import torch.nn.functional as F
from torch import Tensor
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

        # Abridging for readability
        state = batch.states
        action = batch.actions
        reward = batch.rewards
        next_state = batch.next_states
        terminated = batch.terminateds

        # Compute target action
        target_action: Tensor = self._target_policy(next_state)
        # Target policy smoothing: add clipped noise to the target action
        target_action += (
            target_action.clone()
            .normal_(0, self._smoothing_noise_stddev)
            .clamp_(-self._smoothing_noise_clip, self._smoothing_noise_clip)
        )
        # Target action is clipped to lie in valid action range
        target_action.clamp_(-1, 1)

        TD_target = (
            reward
            + ~terminated * self._discount_factor
            # Clipped double-Q learning
            * torch.min(
                *[
                    target_critic(next_state, target_action)
                    for target_critic in self._target_critics
                ]
            )
        )
        action_values = [critic(state, action) for critic in self._critics]

        critic_loss = torch.add(
            *[F.mse_loss(TD_target, action_value) for action_value in action_values]
        )
        [critic_optimiser.zero_grad() for critic_optimiser in self._critic_optimisers]  # type: ignore
        critic_loss.backward()
        [critic_optimiser.step() for critic_optimiser in self._critic_optimisers]

        # "Delayed" policy updates
        if next(self._count) % self._policy_delay == 0:

            # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
            policy_loss: Tensor = -self._critics[0](state, self._policy(state)).mean()
            self._policy_optimiser.zero_grad()
            policy_loss.backward()
            self._policy_optimiser.step()

            # Update frozen target networks by Polyak averaging
            with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
                for critic, target_critic in zip(self._critics, self._target_critics):
                    for 𝜙, 𝜙_targ in zip(
                        critic.parameters(), target_critic.parameters()
                    ):
                        𝜙_targ.mul_(self._polyak)
                        𝜙_targ.add_((1.0 - self._polyak) * 𝜙)
                for 𝜃, 𝜃_targ in zip(
                    self._policy.parameters(), self._target_policy.parameters()
                ):
                    𝜃_targ.mul_(self._polyak)
                    𝜃_targ.add_((1.0 - self._polyak) * 𝜃)

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
