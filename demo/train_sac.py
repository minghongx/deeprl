import math
from functools import partial

import gymnasium as gym
import torch
import torch.optim as optim
import wandb

from deeprl.actor_critic_methods import SAC
from deeprl.actor_critic_methods.neural_network import mlp
from deeprl.actor_critic_methods.experience_replay import UER


def train() -> None:
    device = torch.device("cuda:1")

    # env = gym.make("HalfCheetah-v4")
    env = gym.make("InvertedDoublePendulum-v4")
    obs_dim = math.prod(env.observation_space.shape)
    action_dim = math.prod(env.action_space.shape)

    agent = SAC.init(
        mlp.GaussianPolicy.init(obs_dim, action_dim, [256, 256]),
        mlp.Quality.init(obs_dim, action_dim, [256, 256]),
        partial(optim.Adam, lr=3e-4),
        partial(optim.Adam, lr=3e-4),
        partial(optim.Adam, lr=3e-4),
        UER(1_000_000),
        256,
        0.99,
        -action_dim,
        5e-3,
        device=device
    )

    run = wandb.init(project="IntegTest")

    for episode in range(100_000):
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

        run.log({
            "episodic_return": episodic_return,
        })

        # Checkpointing
        # torch.onnx.export(agent._policy.eval(), state, "policy.onnx"); agent._policy.train()


if __name__ == '__main__':
    train()
