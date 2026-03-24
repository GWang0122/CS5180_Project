"""On-policy rollout buffer with GAE, shared by both algorithms."""

import torch
import numpy as np


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Generalized Advantage Estimation.

    Args:
        rewards:    (T,) tensor of rewards.
        values:     (T,) tensor of value estimates V(s_t).
        dones:      (T,) tensor of done flags (1.0 = episode ended).
        next_value: scalar V(s_{T+1}) bootstrap value.
        gamma:      discount factor.
        gae_lambda: GAE lambda.

    Returns:
        advantages: (T,) tensor.
        returns:    (T,) tensor (advantages + values).
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Stores a fixed-length rollout and converts it to tensors for PPO updates.

    Designed for recurrent policies: stores data in temporal order so hidden
    states can be reconstructed by replaying sequences.
    """

    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def store(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def get(self, next_value: float, gamma: float, gae_lambda: float,
            device: torch.device):
        """Finalize the rollout: compute GAE and return all data as tensors.

        Args:
            next_value: V(s_{T+1}) for bootstrapping.
            gamma: discount factor.
            gae_lambda: GAE lambda.
            device: torch device.

        Returns:
            dict of tensors with keys: observations, actions, log_probs,
            values, advantages, returns.
        """
        obs = torch.tensor(np.array(self.observations), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)

        advantages, returns = compute_gae(
            rewards, values, dones,
            next_value=torch.tensor(next_value, dtype=torch.float32, device=device),
            gamma=gamma, gae_lambda=gae_lambda,
        )

        return {
            "observations": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def __len__(self):
        return len(self.rewards)
