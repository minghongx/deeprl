import numpy as np
from torch import Tensor

from .base import ExperienceReplay, Experience, Batch
from ...data_structures import RotatingList


class UER(ExperienceReplay):
    """
    Uniformly sampled

    In SRS, each subset of k individuals has the same 
    probability of being chosen for the sample as any 
    other subset of k individuals.
    https://en.wikipedia.org/wiki/Simple_random_sample
    """

    def __init__(self, capacity: int) -> None:
        self._buffer = RotatingList[Experience](capacity)
        self._rng = np.random.default_rng()  # https://stackoverflow.com/questions/61676156/how-to-use-the-new-numpy-random-number-generator

    def push(self,
            observation     : Tensor,
            action          : Tensor,
            reward          : Tensor,
            next_observation: Tensor,
            terminated      : Tensor
    ) -> None:
        self._buffer.store( Experience(observation, action, reward, next_observation, terminated) )

    def sample(self, batch_size: int) -> Batch:
        """
        https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        https://stackoverflow.com/a/62951059/20015297
        https://www.pythondoeswhat.com/2015/07/collectionsdeque-random-access-is-on.html
        """
        indices = self._rng.choice(len(self._buffer), batch_size, replace=False)
        experiences = [self._buffer[index] for index in indices]
        return Batch(experiences)
