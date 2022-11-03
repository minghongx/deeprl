from pathlib import Path
from datetime import datetime
from functools import partial

import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# TODO: Load from a file

# env_name = 'MountainCarContinuous-v0'
# env_name = 'HalfCheetah-v4'
# env_name = 'Pendulum-v1'
env_name = 'InvertedDoublePendulum-v4'
# env_name = 'Humanoid-v4'
# env_name = 'Ant-v4'
env = gym.make(env_name)

num_episodes = 10_000
state_dim = env.observation_space.shape[0]  # Rename observation to state for consistency
action_dim = env.action_space.shape[0]
hidden_dims = [400, 300]
actor_lr  = 1e-4
critic_lr = 1e-3
weight_decay = 1e-5
memory_capacity = int(1e6)
batch_size = 2**10
discount_factor = 0.99
polyak = 0.99
clip_bound = 0.5
stddev = 0.2
gaussian_action_noise_stddev = 0.1
gaussian_action_noise_decay_constant = 1e-9
from deeprl.actor_critic_arch.ddpg import DDPG
from deeprl.actor_critic_arch.td3 import TD3
from deeprl.actor_critic_arch.neural_network.mlp import Actor, Critic
from deeprl.actor_critic_arch.experience_replay.uer import UER
from deeprl.actor_critic_arch.experience_replay.per import PER
from deeprl.actor_critic_arch.noise_injection.action_space import Gaussian
from deeprl.actor_critic_arch.noise_injection.parameter_space import AdaptiveParameterNoise
cuda = torch.device(1)

writer = SummaryWriter(log_dir=Path.cwd()/'.logs'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}')
checkpoint_dir = Path.cwd()/'.checkpoints'/'TD3'/f'{env.spec.name}-v{env.spec.version}'/f'{datetime.now().strftime("%Y%m%d%H%M")}'

# agent = DDPG(
#     Actor (state_dim, action_dim, hidden_dims, 'relu', 'tanh').to(cuda),
#     Critic(state_dim, action_dim, hidden_dims, 'relu').to(cuda),
#     partial(optim.Adam, lr=actor_lr , weight_decay=weight_decay),
#     partial(optim.Adam, lr=critic_lr, weight_decay=weight_decay),
#     UER(memory_capacity),
#     # PER(memory_capacity, 0.7),
#     batch_size,
#     discount_factor,
#     polyak,
#     Gaussian(gaussian_action_noise_stddev, gaussian_action_noise_decay_constant)
#     # AdaptiveParameterNoise(0.01, 0.01, 0.999)
# )

agent = TD3(
    Actor (state_dim, action_dim, hidden_dims, 'relu', 'tanh').to(cuda),
    Critic(state_dim, action_dim, hidden_dims, 'relu').to(cuda),
    partial(optim.Adam, lr=actor_lr , weight_decay=weight_decay),
    partial(optim.Adam, lr=critic_lr, weight_decay=weight_decay),
    UER(memory_capacity),
    batch_size,
    discount_factor,
    polyak,
    Gaussian(gaussian_action_noise_stddev, gaussian_action_noise_decay_constant),
    clip_bound,
    stddev
)

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, device=cuda, dtype=torch.float32)
    cumulative_reward = torch.zeros(1, device=cuda)

    for step in range(env.spec.max_episode_steps):
        # Compute action
        # TODO: Improve exploration
        action = agent.compute_action(state)

        # Perform an action
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        cumulative_reward += reward
        # Convert to size(1,) tensor
        next_state = torch.tensor(next_state  , device=cuda, dtype=torch.float32)
        reward     = torch.tensor([reward]    , device=cuda, dtype=torch.float32)
        terminated = torch.tensor([terminated], device=cuda, dtype=torch.bool)

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

writer.close()
