from pathlib import Path
from functools import partial
from datetime import datetime

import torch
import torch.optim as optim
from pettingzoo.mpe import simple_speaker_listener_v3, simple_reference_v2
# Truncation/Termination/Rendering API Update
# https://github.com/Farama-Foundation/PettingZoo/issues/769
from toolz import valmap, merge_with
from torch.utils.tensorboard import SummaryWriter

from deeprl.multi_agent.maddpg.algo import MADDPG, Agent
from deeprl.multi_agent.maddpg.nn import Actor, Critic
from deeprl.multi_agent.maddpg.er import UER


env = simple_speaker_listener_v3.parallel_env(max_cycles=25, continuous_actions=True)  # cooperative communication
# env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=25, continuous_actions=True)
env.reset()  # https://github.com/Farama-Foundation/PettingZoo/issues/640

# TODO: Load from a file
num_episodes = 10_000_000
# num_episodes = 2_000
# max_episode_steps = 1000
hidden_dims = [64, 64]
lr_actor  = 1e-2
lr_critic = 1e-2
memory_capacity = 1_000_000
batch_size = 2**10
discount_factor = 0.95
polyak = 0.99

cuda = torch.device(1)
writer = SummaryWriter(log_dir=Path.cwd()/'.logs'/'MADDPG'/f'{env.metadata["name"]}'/f'{datetime.now().strftime("%Y%m%d%H%M")}')

agents = {
    agent_id: Agent(
        Actor(
            env.observation_space(agent_id).shape[0],
            env.action_space(agent_id).shape[0],
            hidden_dims,
            'relu',
            'softmax').to(cuda),
        Critic(
            sum([env.observation_space(id).shape[0] for id in env.agents]),
            sum([env.action_space(id).shape[0] for id in env.agents]),
            hidden_dims,
            'relu').to(cuda),
        partial(optim.Adam, lr=lr_actor ),
        partial(optim.Adam, lr=lr_critic),
        discount_factor,
        polyak
    ) for agent_id in env.agents }
maddpg = MADDPG(agents, UER(memory_capacity), batch_size)

for episode in range(num_episodes):
    observation = env.reset()
    observation = valmap(partial(torch.tensor, device=cuda), observation)
    cumulative_reward = list()

    # for step in range(max_episode_steps):
    while True:
        action = {agent_id: maddpg.compute_action(agent_id, obs) for agent_id, obs in observation.items()}
        next_observation, reward, terminated, truncated, _ = env.step( valmap(lambda tensor: tensor.cpu().numpy(), action) )
        cumulative_reward.append(reward)

        # Convert to size(1,) tensor
        next_observation = valmap(partial(torch.tensor, device=cuda), next_observation)
        reward     = valmap(lambda r: torch.tensor([r], device=cuda), reward)
        terminated = valmap(lambda t: torch.tensor([t], device=cuda), terminated)

        maddpg.step(observation, action, reward, next_observation, terminated)

        if not env.agents:
            break
        # Move to the next observations
        observation = next_observation

    # Logging
    writer.add_scalars(f'{env.metadata["name"]}/cumulative_reward', merge_with(sum, cumulative_reward), episode)

writer.close()
