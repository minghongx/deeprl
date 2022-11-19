from copy import deepcopy
# from collections.abc import Callable, Iterator, Mapping
from typing import Callable, Iterator, Mapping  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
# from pettingzoo.utils.env import AgentID
AgentID = str

from .nn import Actor, Critic
from .er import ExperienceReplay


class Agent:

    def __init__(self,
            policy: Actor,
            critic: Critic,
            policy_optimiser: Callable[[Iterator[Parameter]], Optimizer],  # TODO: Adopt PEP677 (Iterator[Parameter]) -> Optimizer
            critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
            discount_factor: float,
            polyak: float,
    ) -> None:

        self.policy = policy
        self.critic = critic

        self.target_policy = deepcopy(self.policy)
        self.target_critic = deepcopy(self.critic)
        # Freeze target networks with respect to optimisers (only update via Polyak averaging)
        self.target_policy.requires_grad_(False)
        self.target_critic.requires_grad_(False)

        self.policy_optimiser = policy_optimiser(self.policy.parameters())
        self.critic_optimiser = critic_optimiser(self.critic.parameters())

        self.discount_factor = discount_factor
        self.polyak = polyak


class MADDPG:

    def __init__(self,
            agents: Mapping[AgentID, Agent],
            experience_replay: ExperienceReplay,
            batch_size: int
    ) -> None:

        self._agents = agents
        self._experience_replay = experience_replay
        self._batch_size = batch_size

    def step(self,
            observation     : Mapping[AgentID, Tensor],
            action          : Mapping[AgentID, Tensor],
            reward          : Mapping[AgentID, Tensor],
            next_observation: Mapping[AgentID, Tensor],
            terminated      : Mapping[AgentID, Tensor]
    ) -> None:

        self._experience_replay.push(observation, action, reward, next_observation, terminated)
        for agent_id in self._agents.keys():
            self._update_main_networks(agent_id)
        for agent_id in self._agents.keys():
            self._update_target_networks(agent_id)

    def _update_main_networks(self, agent_id: AgentID) -> None:

        try:
            batch = self._experience_replay.sample(self._batch_size)
        except ValueError:
            return

        # Abbrivating for readability
        policy           = self._agents[agent_id].policy
        critic           = self._agents[agent_id].critic
        target_critic    = self._agents[agent_id].target_critic
        policy_optimiser = self._agents[agent_id].policy_optimiser
        critic_optimiser = self._agents[agent_id].critic_optimiser
        discount_factor  = self._agents[agent_id].discount_factor
        # Prepare operands
        observation = batch.observations[agent_id]
        reward      = batch.rewards[agent_id]
        terminated  = batch.terminateds[agent_id]
        observation_of_all_agents      = list(batch.observations.values())
        action_of_all_agents           = list(batch.actions.values())
        next_observation_of_all_agents = list(batch.next_observations.values())
        next_action_of_all_agents = [self._agents[id].target_policy(batch.next_observations[id]) for id in self._agents.keys()]

        TD_targets = reward + ~terminated * discount_factor * target_critic(next_observation_of_all_agents, next_action_of_all_agents)

        critic_loss = F.mse_loss( TD_targets, critic(observation_of_all_agents, action_of_all_agents) )
        critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_optimiser.step()

        batch.actions[agent_id] = policy(observation)
        policy_loss = -critic(observation_of_all_agents, list(batch.actions.values())).mean()
        policy_optimiser.zero_grad()
        policy_loss.backward()
        policy_optimiser.step()

    @torch.no_grad()
    def _update_target_networks(self, agent_id: AgentID) -> None:

        # Abbrivating for readability
        policy        = self._agents[agent_id].policy
        critic        = self._agents[agent_id].critic
        target_policy = self._agents[agent_id].target_policy
        target_critic = self._agents[agent_id].target_critic
        polyak        = self._agents[agent_id].polyak

        # Update frozen target networks by Polyak averaging
        for ϕ, ϕ_targ in zip(critic.parameters(), target_critic.parameters()):
            ϕ_targ.mul_(polyak)
            ϕ_targ.add_( (1.0 - polyak) * ϕ )
        for θ, θ_targ in zip(policy.parameters(), target_policy.parameters()):
            θ_targ.mul_(polyak)
            θ_targ.add_( (1.0 - polyak) * θ )

    @torch.no_grad()
    def compute_action(self, agent_id: AgentID, observation: Tensor) -> Tensor:
        return self._agents[agent_id].policy(observation)
