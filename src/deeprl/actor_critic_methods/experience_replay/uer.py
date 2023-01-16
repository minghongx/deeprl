import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union

from ..._data_structures import RotatingList
from ._base import Batch, Experience, ExperienceReplay


class UER(ExperienceReplay):
    """
    Uniformly sampled

    In SRS, each subset of k individuals has the same
    probability of being chosen for the sample as any
    other subset of k individuals.
    https://en.wikipedia.org/wiki/Simple_random_sample
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self._buffer = RotatingList[Experience](capacity)
        self._batch = Batch.preallocate(batch_size, state_dim, action_dim, device)
        self._rng = np.random.default_rng()

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        terminated: Tensor,
    ) -> None:
        self._buffer.store(Experience(state, action, reward, next_state, terminated))

    def sample(self) -> Batch:
        """
        https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        https://stackoverflow.com/a/62951059/20015297
        https://www.pythondoeswhat.com/2015/07/collectionsdeque-random-access-is-on.html
        """
        indices = self._rng.choice(len(self._buffer), self._batch.size, replace=False)
        experiences = [self._buffer[index] for index in indices]
        return self._batch.of(experiences)
