import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Linear, ModuleList

from deeprl.actor_critic_methods.neural_network import mlp


def test_gaussian_policy_mode():
    randint = random.randint(1, 10)
    policy = mlp.TanhGaussianPolicy(
        ModuleList([Linear(randint, randint) for _ in range(randint)]),
        Linear(randint, randint),
        Linear(randint, randint),
        F.relu,
    )
    state = torch.rand(randint)

    evaluation_mode = policy.eval()
    assert isinstance(evaluation_mode(state), Tensor)

    training_mode = policy.train()
    assert isinstance(training_mode(state), Distribution)
