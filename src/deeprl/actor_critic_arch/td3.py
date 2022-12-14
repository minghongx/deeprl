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
        clip_bound: float,
        stddev: float,
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
        self._clip_bound = clip_bound
        self._stddev = stddev
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
        self._update()

    def _update(self) -> None:

        try:
            batch = self._experience_replay.sample(self._batch_size)
        except ValueError:
            return

        # Compute target actions
        target_actions: Tensor = self._target_policy(batch.next_states)
        # Add clipped noise
        target_actions += (
            target_actions.clone()
            .normal_(0, self._stddev)
            .clamp_(-self._clip_bound, self._clip_bound)
        )
        # Target actions are clipped to lie in valid action range
        target_actions.clamp_(-1, 1)

        TD_targets = (
            batch.rewards
            + ~batch.terminateds
            * self._discount_factor
            * torch.min(
                *[
                    target_critic(batch.next_states, target_actions)
                    for target_critic in self._target_critics
                ]
            )
        )
        ls_action_values = [
            critic(batch.states, batch.actions) for critic in self._critics
        ]

        critic_loss = torch.add(
            *[
                F.mse_loss(TD_targets, action_values)
                for action_values in ls_action_values
            ]
        )
        [critic_optimiser.zero_grad() for critic_optimiser in self._critic_optimisers]
        critic_loss.backward()
        [critic_optimiser.step() for critic_optimiser in self._critic_optimisers]

        if next(self._count) % self._policy_delay == 0:

            # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
            policy_loss: Tensor = -self._critics[0](
                batch.states, self._policy(batch.states)
            ).mean()
            self._policy_optimiser.zero_grad()
            policy_loss.backward()
            self._policy_optimiser.step()

            # Update frozen target networks by Polyak averaging
            with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
                for critic, target_critic in zip(self._critics, self._target_critics):
                    for ϕ, ϕ_targ in zip(
                        critic.parameters(), target_critic.parameters()
                    ):
                        ϕ_targ.mul_(self._polyak)
                        ϕ_targ.add_((1.0 - self._polyak) * ϕ)
                for θ, θ_targ in zip(
                    self._policy.parameters(), self._target_policy.parameters()
                ):
                    θ_targ.mul_(self._polyak)
                    θ_targ.add_((1.0 - self._polyak) * θ)

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
