from dataclasses import dataclass, fields
from typing import Iterator, Sequence

import torch
from attrs import define, field
from torch import Tensor


@dataclass
class Experience:
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    terminated: Tensor

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.values())


@define(slots=False)
class Batch:
    experiences: Sequence[Experience]
    states: Tensor = field(init=False)
    actions: Tensor = field(init=False)
    rewards: Tensor = field(init=False)
    next_states: Tensor = field(init=False)
    terminateds: Tensor = field(init=False)

    def __attrs_post_init__(self) -> None:
        for attr, unstacked in zip(fields(Experience), zip(*self.experiences)):
            setattr(self, f"{attr.name}s", torch.stack(unstacked))
