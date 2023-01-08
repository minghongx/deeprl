from copy import deepcopy

import torch
import torch.nn as nn
from cytoolz import comp
from torch import Tensor

from ..neural_network import DeterministicActor


class AdaptiveParameterNoise:
    def __init__(
        self, stddev: float, desired_stddev: float, adoption_coefficient: float
    ) -> None:
        self.stddev = stddev
        self.desired_stddev = desired_stddev
        self.adoption_coefficient = adoption_coefficient

    @torch.no_grad()
    def perturb(self, policy: DeterministicActor) -> DeterministicActor:
        perturbed_policy = deepcopy(policy)
        perturbed_policy.requires_grad_(False)
        perturbed_policy.apply(self._add_gaussian_noise_to_weights)
        return perturbed_policy

    @torch.no_grad()
    def _add_gaussian_noise_to_weights(self, m: nn.Module) -> None:
        if hasattr(m, "weight"):
            m.weight.add_(  # FIXME: Type
                torch.randn(m.weight.size(), device=m.weight.device) * self.stddev  # type: ignore
            )

    @torch.no_grad()
    def adapt(self, action: Tensor, perturbed_action: Tensor) -> None:
        stddev = comp(torch.sqrt, torch.mean, torch.square)(action - perturbed_action)
        # TODO: Avaliable since version 3.10. See PEP 634
        # match stddev:
        # case _ if stddev > self.desired_stddev:
        # case _ if stddev < self.desired_stddev:
        if stddev > self.desired_stddev:
            self.stddev *= self.adoption_coefficient
        elif stddev < self.desired_stddev:
            self.stddev /= self.adoption_coefficient
