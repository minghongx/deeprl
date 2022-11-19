# https://github.com/pfnet/pfrl/blob/master/pfrl/collections/prioritized.py

from typing import TypeVar, Generic, Sized
from typing import Tuple  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.

import numpy as np

from .rotating_list import RotatingList


_T = TypeVar('_T')


class SumTree(Sized, Generic[_T]):

    def __init__(self, capacity: int) -> None:
        self._weights = np.zeros(capacity * 2 - 1)
        self._leaves  = RotatingList[_T](capacity)
        self._bias    = len(self._weights) - capacity

    def retrieve(self, value: float) -> Tuple[int, _T]:
        parent = 0
        while True:
            left  = parent * 2 + 1
            right = left + 1
            try:
                left_weight = self._weights[left]
            except IndexError:
                leaf = parent
                break
            if value < left_weight or np.isclose(value, left_weight):
                parent = left
            else:
                value -= left_weight
                parent = right
        leaf = min(leaf, len(self._leaves) + self._bias - 1)
        return leaf, self._leaves[leaf - self._bias]

    def store(self, __object: _T, priority) -> int:
        leaf = self._leaves.store(__object) + self._bias
        self.update_priority(leaf, priority)
        return leaf

    def update_priority(self, leaf: int, priority: float) -> None:
        change = priority - self._weights[leaf]
        node = leaf
        while True:
            self._weights[node] += change
            if node == 0:  # reached the root node
                break
            node = (node - 1) // 2  # moves to the parent node

    def __len__(self) -> int:
        return len(self._leaves)
