import math
from functools import partial

import gymnasium as gym
import hydra
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf

from conf import EnvConfig, TD3Config
from deeprl.actor_critic_methods import TD3
from deeprl.actor_critic_methods.neural_network import mlp
from deeprl.actor_critic_methods.experience_replay import UER
from deeprl.actor_critic_methods.noise_injection.action_space import Gaussian

from torch.profiler import profile, ProfilerActivity

@hydra.main(version_base=None, config_path='conf', config_name='train_td3')
def train(cfg: DictConfig) -> None:
    env_cfg = EnvConfig(**cfg['env'])
    td3_cfg = TD3Config(**cfg['td3'])

    device = torch.device(env_cfg.device)

    env = gym.make(env_cfg.name)
    obs_dim = math.prod(env.observation_space.shape)
    action_dim = math.prod(env.action_space.shape)

    agent = TD3.init(
        mlp.Policy.init(obs_dim, action_dim, td3_cfg.hidden_dims),
        mlp.Quality.init(obs_dim, action_dim, td3_cfg.hidden_dims),
        partial(optim.Adam, lr=td3_cfg.actor_lr),
        partial(optim.Adam, lr=td3_cfg.critic_lr),
        UER(td3_cfg.memory_capacity),
        td3_cfg.batch_size,
        td3_cfg.discount_factor,
        td3_cfg.target_smoothing_factor,
        Gaussian(td3_cfg.action_noise_stdev),
        td3_cfg.smoothing_noise_stdev,
        td3_cfg.smoothing_noise_clip,
        device=device,
    )

    run = wandb.init(project="IntegTest", config=OmegaConf.to_container(cfg, resolve=True))

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(
    #         skip_first=2048, wait=10, warmup=13, active=27, repeat=4),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./traces'),
    #     profile_memory=True,
    #     with_stack=True,
    # ) as profiler:
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

            # profiler.step()
            if terminated or truncated:
                break
            # Move to the next state
            state = next_state

        run.log({
            "episodic_return": episodic_return,
        })

        # Checkpointing
        # torch.onnx.export(agent._policy, state, "policy.onnx")


if __name__ == '__main__':
    train()
