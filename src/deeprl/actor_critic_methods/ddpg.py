from copy import deepcopy

# from collections.abc import Callable, Iterator
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Callable,
    Iterator,
    Union,
)

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .experience_replay import PER, ExperienceReplay
from .neural_network import ActionCritic, DeterministicActor
from .noise_injection.action_space import ActionNoise
from .noise_injection.parameter_space import AdaptiveParameterNoise


class DDPG:
    def __init__(
        self,
        policy: DeterministicActor,
        critic: ActionCritic,
        policy_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
        experience_replay: ExperienceReplay,
        batch_size: int,
        discount_factor: float,
        polyak: float,
        policy_noise: Union[ActionNoise, AdaptiveParameterNoise, None],
    ) -> None:

        self._policy = policy
        self._critic = critic
        self._target_policy = deepcopy(self._policy)
        self._target_critic = deepcopy(self._critic)
        # Freeze target networks with respect to optimisers (only update via Polyak averaging)
        self._target_policy.requires_grad_(False)
        self._target_critic.requires_grad_(False)

        self._policy_optimiser = policy_optimiser(self._policy.parameters())
        self._critic_optimiser = critic_optimiser(self._critic.parameters())

        self._experience_replay = experience_replay
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._polyak = polyak
        self._policy_noise = policy_noise

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

        TD_targets = (
            batch.rewards
            + ~batch.terminateds
            * self._discount_factor
            * self._target_critic(
                batch.next_states, self._target_policy(batch.next_states)
            )
        )
        action_values = self._critic(batch.states, batch.actions)

        critic_loss = F.mse_loss(TD_targets, action_values)
        self._critic_optimiser.zero_grad()
        critic_loss.backward()
        self._critic_optimiser.step()

        # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
        policy_loss: Tensor = -self._critic(
            batch.states, self._policy(batch.states)
        ).mean()
        self._policy_optimiser.zero_grad()
        policy_loss.backward()
        self._policy_optimiser.step()

        # Update frozen target networks by Polyak averaging
        with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
            for ϕ, ϕ_targ in zip(
                self._critic.parameters(), self._target_critic.parameters()
            ):
                ϕ_targ.mul_(self._polyak)
                ϕ_targ.add_((1.0 - self._polyak) * ϕ)
            for θ, θ_targ in zip(
                self._policy.parameters(), self._target_policy.parameters()
            ):
                θ_targ.mul_(self._polyak)
                θ_targ.add_((1.0 - self._polyak) * θ)

            if isinstance(self._experience_replay, PER):
                TD_errors = TD_targets - action_values
                priorities = torch.abs(TD_errors).cpu().numpy()
                setattr(batch, "priorities", priorities)
                self._experience_replay.update_priorities(batch)

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        action: Tensor = self._policy(state)
        # TODO: Avaliable since version 3.10. See PEP 634
        # match self._policy_noise:
        #     case ActionNoise():
        #     case AdaptiveParameterNoise():
        #     case _:
        if isinstance(self._policy_noise, ActionNoise):
            noise = self._policy_noise(action)
            action = (action + noise).clamp(-1, 1)
        if isinstance(self._policy_noise, AdaptiveParameterNoise):
            perturbed_policy = self._policy_noise.perturb(self._policy)
            perturbed_action = perturbed_policy(state)
            self._policy_noise.adapt(action, perturbed_action)
            action = perturbed_action
        return action
