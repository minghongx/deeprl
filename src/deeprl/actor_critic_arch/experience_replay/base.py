from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor
from attrs import define, field


@dataclass
class Experience:
    state     : Tensor
    action    : Tensor
    reward    : Tensor
    next_state: Tensor
    terminated: Tensor

    def __iter__(self):
        return iter(self.__dict__.values())


@define(slots=False)  # https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
class Batch:
    experiences: Sequence[Experience]
    states     : Tensor = field(init=False)
    actions    : Tensor = field(init=False)
    rewards    : Tensor = field(init=False)
    next_states: Tensor = field(init=False)
    terminateds: Tensor = field(init=False)

    def __attrs_post_init__(self) -> None:
        for field, unstacked in zip( fields(Experience), zip(*self.experiences) ):
            setattr(self, field.name + 's', torch.stack(unstacked))


class ExperienceReplay(ABC):

    @abstractmethod
    def push(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, terminated: Tensor) -> None:
        ...

    # TODO: https://docs.python.org/3/library/typing.html#typing.overload
    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        ...
