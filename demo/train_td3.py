import math
from datetime import datetime
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch
import torch.optim as optim
import hydra
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from conf import EnvConfig, TD3Config
from deeprl.actor_critic_methods import TD3
from deeprl.actor_critic_methods.neural_network.mlp import Actor, Critic
from deeprl.actor_critic_methods.experience_replay import UER
from deeprl.actor_critic_methods.noise_injection.action_space import Gaussian


@hydra.main(version_base=None, config_path='conf', config_name='train_td3')
def train(cfg: DictConfig) -> None:
    env_cfg = EnvConfig(**cfg['env'])
    td3_cfg = TD3Config(**cfg['td3'])

    env = gym.make(env_cfg.gym_name)
    device = torch.device(env_cfg.device)

    agent = TD3(
        device,
        math.prod(env.observation_space.shape),
        math.prod(env.action_space.shape),
        partial(Actor, hidden_dims=td3_cfg.hidden_dims),
        partial(Critic, hidden_dims=td3_cfg.hidden_dims),
        partial(optim.Adam, lr=td3_cfg.actor_lr , weight_decay=td3_cfg.weight_decay),
        partial(optim.Adam, lr=td3_cfg.critic_lr, weight_decay=td3_cfg.weight_decay),
        UER(td3_cfg.memory_capacity),
        td3_cfg.batch_size,
        td3_cfg.discount_factor,
        td3_cfg.target_smoothing_factor,
        Gaussian(td3_cfg.action_noise_stddev, td3_cfg.action_noise_decay_const),
        td3_cfg.smoothing_noise_stddev,
        td3_cfg.smoothing_noise_clip,
    )

    checkpoint_dir = Path(__file__).resolve().parent/'.checkpoints'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}'
    with SummaryWriter(log_dir=Path(__file__).resolve().parent/'.logs'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}') as writer:
        for episode in range(env_cfg.num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32)
            episodic_return = torch.zeros(1, device=device)

            while True:
                action = agent.compute_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                next_state = torch.tensor(next_state  , device=device, dtype=torch.float32)
                # Convert to size(1,) tensor
                reward     = torch.tensor([reward]    , device=device, dtype=torch.float32)
                terminated = torch.tensor([terminated], device=device, dtype=torch.bool)

                episodic_return += reward
                # Store a transition in the experience replay and perform one step of the optimisation
                agent.step(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    break
                # Move to the next state
                state = next_state

            # Logging
            writer.add_scalar(f'{env.spec.name}-v{env.spec.version}/episodic_return', episodic_return.item(), episode)

            # Periodic checkpointing
            if episode % 20 == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy_scripted = torch.jit.script(agent._policy)
                policy_scripted.save(checkpoint_dir/f'ep{episode}.pt')


if __name__ == '__main__':
    train()
