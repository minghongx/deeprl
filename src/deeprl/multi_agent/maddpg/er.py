from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

# from collections.abc import Mapping, MutableMapping
from typing import (  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
    Iterator,
    List,
    Mapping,
    MutableMapping,
)

import numpy as np
import torch
from attrs import define, field
from cytoolz import merge_with
from torch import Tensor

# from pettingzoo.utils.env import AgentID
AgentID = str

from ...data_structures.rotating_list import RotatingList  # noqa: E402


@dataclass
class Experience:
    observation: Mapping[AgentID, Tensor]
    action: Mapping[AgentID, Tensor]
    reward: Mapping[AgentID, Tensor]
    next_observation: Mapping[AgentID, Tensor]
    terminated: Mapping[AgentID, Tensor]

    def __iter__(self) -> Iterator:
        """https://stackoverflow.com/a/70753113/20015297"""
        return iter(self.__dict__.values())


@define
class Batch:
    experiences: List[Experience]
    observations: Mapping[AgentID, Tensor] = field(init=False)
    actions: MutableMapping[AgentID, Tensor] = field(init=False)
    rewards: Mapping[AgentID, Tensor] = field(init=False)
    next_observations: Mapping[AgentID, Tensor] = field(init=False)
    terminateds: Mapping[AgentID, Tensor] = field(init=False)

    def __attrs_post_init__(self) -> None:
        for field, unstacked in zip(fields(Experience), zip(*self.experiences)):
            setattr(self, field.name + "s", merge_with(torch.stack, unstacked))


class ExperienceReplay(ABC):
    @abstractmethod
    def push(
        self,
        observation: Mapping[AgentID, Tensor],
        action: Mapping[AgentID, Tensor],
        reward: Mapping[AgentID, Tensor],
        next_observation: Mapping[AgentID, Tensor],
        terminated: Mapping[AgentID, Tensor],
    ) -> None:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        ...


class UER(ExperienceReplay):
    def __init__(self, capacity: int) -> None:
        self._buffer = RotatingList[Experience](capacity)
        self._rng = np.random.default_rng()

    def push(
        self,
        observation: Mapping[AgentID, Tensor],
        action: Mapping[AgentID, Tensor],
        reward: Mapping[AgentID, Tensor],
        next_observation: Mapping[AgentID, Tensor],
        terminated: Mapping[AgentID, Tensor],
    ) -> None:
        self._buffer.store(
            Experience(observation, action, reward, next_observation, terminated)
        )

    def sample(self, batch_size: int) -> Batch:
        indices = self._rng.choice(len(self._buffer), batch_size, replace=False)
        experiences = [self._buffer[index] for index in indices]
        return Batch(experiences)
