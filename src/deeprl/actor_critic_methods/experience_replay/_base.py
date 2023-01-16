from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import torch
from attrs import define, field, setters
from torch import Tensor


@define
class Experience:
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    terminated: Tensor


@define(slots=False)
class Batch:
    size: int = field(on_setattr=setters.frozen)
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    terminateds: Tensor

    @classmethod
    def preallocate(
        cls,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "Batch":
        states = torch.empty(batch_size, state_dim, device=device)
        actions = torch.empty(batch_size, action_dim, device=device)
        rewards = torch.empty(batch_size, 1, device=device)
        next_states = torch.empty(batch_size, state_dim, device=device)
        terminateds = torch.empty(batch_size, 1, device=device, dtype=torch.bool)
        return cls(batch_size, states, actions, rewards, next_states, terminateds)

    def of(self, experiences: Sequence[Experience]) -> "Batch":
        for i, exp in enumerate(experiences):
            self.states[i] = exp.state
            self.actions[i] = exp.action
            self.rewards[i] = exp.reward
            self.next_states[i] = exp.next_state
            self.terminateds[i] = exp.terminated
        return self

    def to(self, device: Optional[Union[torch.device, str]]) -> "Batch":
        self.states.to(device)
        self.actions.to(device)
        self.rewards.to(device)
        self.next_states.to(device)
        self.terminateds.to(device)
        return self


class ExperienceReplay(ABC):
    @abstractmethod
    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        terminated: Tensor,
    ) -> None:
        ...

    # TODO: https://docs.python.org/3/library/typing.html#typing.overload
    @abstractmethod
    def sample(self) -> Batch:
        ...
