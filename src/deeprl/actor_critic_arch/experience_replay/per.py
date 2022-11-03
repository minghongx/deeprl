import random

import numpy as np
from torch import Tensor

from .base import ExperienceReplay, Experience, Batch
from ...data_structures.sum_tree import SumTree


class PER(ExperienceReplay):
    """Prioritised"""

    def __init__(self, capacity: int, α: float, ϵ: float = 0.01) -> None:
        self._buffer = SumTree[Experience](capacity)
        self._α = α
        self._ϵ = ϵ
        self._maximal_priority = ϵ

    def push(self,
        observation     : Tensor,
        action          : Tensor,
        reward          : Tensor,
        next_observation: Tensor,
        terminated      : Tensor
    ) -> None:
        self._buffer.store(
            Experience(observation, action, reward, next_observation, terminated),
            self._maximal_priority)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > len(self._buffer):
            raise ValueError
        bounds = np.linspace(0, self._buffer._weights[0], batch_size + 1)
        indices, experiences = zip(*[self._buffer.retrieve( random.uniform(l, h) ) for l, h in zip(bounds, bounds[1:])])
        batch = Batch(experiences)
        setattr(batch, 'indices', indices)
        return batch

    def update_priorities(self, batch: Batch):
        if not hasattr(batch, 'indices') or not hasattr(batch, 'priorities'):
            raise ValueError('Missing attribute "indices" or "priorities".')
        for idx, priority in zip( getattr(batch, 'indices'), getattr(batch, 'priorities') ):
            p = (priority + self._ϵ) ** self._α
            self._buffer.update_priority(idx, p)
            self._maximal_priority = max(self._maximal_priority, p)
