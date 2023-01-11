from typing import List  # TODO: Deprecated since version 3.9. See Generic Alias Type and PEP 585.

from pydantic import BaseModel, validator
from omegaconf import OmegaConf


class EnvConfig(BaseModel):
    name: str
    num_episodes: int
    device: str


class TD3Config(BaseModel):
    hidden_dims: List[int]
    actor_lr : float
    critic_lr: float
    memory_capacity: int
    batch_size: int
    discount_factor: float
    target_smoothing_factor: float
    smoothing_noise_stddev: float
    smoothing_noise_clip: float
    action_noise_stddev: float

    def __init__(self, **data):
        data['hidden_dims'] = OmegaConf.to_object(data['hidden_dims'])
        super().__init__(**data)

    @validator('target_smoothing_factor')
    def is_between_0_and_1(cls, v):
        assert 0 <= v <= 1
        return v
