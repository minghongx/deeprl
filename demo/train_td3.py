from pathlib import Path
from datetime import datetime
from functools import partial

import gymnasium as gym
import torch
import torch.optim as optim
import hydra
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from .conf import EnvConfig, TD3Config
from deeprl.actor_critic_arch.td3 import TD3
from deeprl.actor_critic_arch.neural_network.mlp import Actor, Critic
from deeprl.actor_critic_arch.experience_replay.uer import UER
from deeprl.actor_critic_arch.noise_injection.action_space import Gaussian


@hydra.main(version_base=None, config_path='conf', config_name='train_td3')
def train(cfg: DictConfig) -> None:
    env_conf = EnvConfig(**cfg['env'])
    td3_conf = TD3Config(**cfg['td3'])

    env = gym.make(env_conf.gym_name)
    device = torch.device(env_conf.device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TD3(
        Actor (state_dim, action_dim, td3_conf.hidden_dims, 'relu', 'tanh').to(device),
        Critic(state_dim, action_dim, td3_conf.hidden_dims, 'relu').to(device),
        partial(optim.Adam, lr=td3_conf.actor_lr , weight_decay=td3_conf.weight_decay),
        partial(optim.Adam, lr=td3_conf.critic_lr, weight_decay=td3_conf.weight_decay),
        UER(td3_conf.memory_capacity),
        td3_conf.batch_size,
        td3_conf.discount_factor,
        td3_conf.polyak,
        Gaussian(td3_conf.action_noise_stddev, td3_conf.action_noise_decay_const),
        td3_conf.clip_bound,
        td3_conf.stddev
    )

    checkpoint_dir = Path.cwd()/'.checkpoints'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}'
    with SummaryWriter(log_dir=Path.cwd()/'.logs'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}') as writer:
        for episode in range(env_conf.num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32)
            cumulative_reward = torch.zeros(1, device=device)

            while True:
                # Compute action
                # TODO: Improve exploration
                action = agent.compute_action(state)

                # Perform an action
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                cumulative_reward += reward
                # Convert to size(1,) tensor
                next_state = torch.tensor(next_state  , device=device, dtype=torch.float32)
                reward     = torch.tensor([reward]    , device=device, dtype=torch.float32)
                terminated = torch.tensor([terminated], device=device, dtype=torch.bool)

                # Store a transition in the experience replay and perform one step of the optimisation
                agent.step(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    break

                # Move to the next state
                state = next_state

            # Logging
            # TODO: Plot mean ± stddev curve for selecting the best model
            writer.add_scalar(f'{env.spec.name}-v{env.spec.version}/cumulative_reward', cumulative_reward.item(), episode)

            if episode % 20 == 0:
                # Checkpointing
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy_scripted = torch.jit.script(agent._policy)
                policy_scripted.save(checkpoint_dir/f'ep{episode}.pt')


if __name__ == '__main__':
    train()
