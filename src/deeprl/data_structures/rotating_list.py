from typing import (
    List,  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.
)
from typing import Union  # TODO: Unnecessary since version 3.10. See PEP 604.
from typing import Generic, Sequence, TypeVar, cast, overload

_T = TypeVar("_T")


class RotatingList(Sequence[_T], Generic[_T]):
    """
    https://stackoverflow.com/a/56171119/20015297
    https://stackoverflow.com/a/2167962/20015297
    """

    def __init__(self, capacity: int) -> None:
        self._list = cast(List[_T], [None for _ in range(capacity)])
        self._capacity = capacity
        self._next_idx: int = 0
        self._size: int = 0

    def store(self, __object: _T) -> int:
        self._list[self._next_idx] = __object
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return idx

    @overload
    def __getitem__(self, idx: int) -> _T:
        ...

    @overload
    def __getitem__(self, idx: slice) -> List[_T]:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[_T, List[_T]]:
        return self._list[idx]

    def __len__(self) -> int:
        return self._size
