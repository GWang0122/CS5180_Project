"""Shared evaluation protocol — identical for both algorithms."""

import torch
import numpy as np
from common.env_wrappers import make_env


@torch.no_grad()
def evaluate(model, env_id: str, num_episodes: int = 10, seed: int = 0,
             max_steps: int = 256, device: torch.device = torch.device("cpu")):
    """Run the policy greedily for num_episodes and return stats.

    Args:
        model: any module that implements:
               - init_hidden(batch_size, device) -> hidden
               - step(obs_tensor, hidden) -> action, log_prob, value, hidden
        env_id: MiniGrid environment ID.
        num_episodes: how many episodes to average over.
        seed: base seed (incremented per episode).
        max_steps: episode step limit.
        device: torch device.

    Returns:
        dict with 'mean_reward', 'std_reward', 'mean_length', 'std_length',
        and 'rewards' / 'lengths' lists.
    """
    model.eval()
    rewards_all = []
    lengths_all = []

    for ep in range(num_episodes):
        env = make_env(env_id, seed=seed + ep, max_steps=max_steps)
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)

        ep_reward = 0.0
        ep_length = 0

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, hidden = model.step(obs_t, hidden)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

            if terminated or truncated:
                break

        env.close()
        rewards_all.append(ep_reward)
        lengths_all.append(ep_length)

    model.train()
    return {
        "mean_reward": float(np.mean(rewards_all)),
        "std_reward": float(np.std(rewards_all)),
        "mean_length": float(np.mean(lengths_all)),
        "std_length": float(np.std(lengths_all)),
        "rewards": rewards_all,
        "lengths": lengths_all,
    }
