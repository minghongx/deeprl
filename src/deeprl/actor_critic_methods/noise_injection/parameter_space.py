from copy import deepcopy

import torch
import torch.nn as nn
from attrs import define
from cytoolz import comp
from torch import Tensor


@define
class AdaptiveParameterNoise:
    stdev: float
    desired_stdev: float
    adoption_coeff: float

    @torch.no_grad()
    def perturb(self, policy: nn.Module) -> nn.Module:
        perturbed_policy = deepcopy(policy).apply(self._add_gaussian_noise_to_weights)
        perturbed_policy.requires_grad_(False)
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
