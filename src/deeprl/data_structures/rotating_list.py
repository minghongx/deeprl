from typing import TypeVar, Sequence, Generic, cast
from typing import List  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.


_T = TypeVar('_T')


class RotatingList(Sequence[_T], Generic[_T]):
    """
    https://stackoverflow.com/a/56171119/20015297
    https://stackoverflow.com/a/2167962/20015297
    """

    def __init__(self, capacity: int) -> None:
        self._list = cast( List[_T], [None for _ in range(capacity)] )  # "cast" is a helper function that overrides the inferred
        self._capacity = capacity
        self._next_idx: int = 0
        self._size : int = 0

    def store(self, __object: _T) -> int:
        self._list[self._next_idx] = __object
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._capacity
        self._size  = min(self._size + 1, self._capacity)
        return idx

    def __getitem__(self, key):
        return self._list[key]

    def __len__(self):
        return self._size
