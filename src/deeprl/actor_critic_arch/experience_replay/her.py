from torch import Tensor

from .base import ExperienceReplay, Experience, Batch
from ...data_structures.sum_tree import SumTree


class HER(ExperienceReplay):

    def __init__(self) -> None:
        ...