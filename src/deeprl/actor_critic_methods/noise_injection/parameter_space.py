from copy import deepcopy

import torch
import torch.nn as nn
from cytoolz import comp
from torch import Tensor

from ..neural_network import DeterministicActor


class AdaptiveParameterNoise:
    def __init__(
        self, stdev: float, desired_stdev: float, adoption_coeff: float
    ) -> None:
        self.stdev = stdev
        self.desired_stdev = desired_stdev
        self.adoption_coeff = adoption_coeff

    @torch.no_grad()
    def perturb(self, policy: DeterministicActor) -> DeterministicActor:
        perturbed_policy = deepcopy(policy)
        perturbed_policy.requires_grad_(False)
        perturbed_policy.apply(self._add_gaussian_noise_to_weights)
        return perturbed_policy

    @torch.no_grad()
    def _add_gaussian_noise_to_weights(self, m: nn.Module) -> None:
        if hasattr(m, "weight"):
            m.weight.add_(torch.randn_like(m.weight) * self.stdev)  # type: ignore

    @torch.no_grad()
    def adapt(self, action: Tensor, perturbed_action: Tensor) -> None:
        stdev = comp(torch.sqrt, torch.mean, torch.square)(action - perturbed_action)
        if stdev > self.desired_stdev:
            self.stdev *= self.adoption_coeff
        elif stdev < self.desired_stdev:
            self.stdev /= self.adoption_coeff
