"""Run a random-action baseline on both environments.

Usage:
    conda activate cs5180_project
    python -m scripts.random_baseline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from common.env_wrappers import make_env


def random_baseline(env_id: str, num_episodes: int = 100, seed: int = 42,
                    max_steps: int = 256):
    rewards = []
    lengths = []

    for ep in range(num_episodes):
        env = make_env(env_id, seed=seed + ep, max_steps=max_steps)
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0

        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1
            if terminated or truncated:
                break

        env.close()
        rewards.append(ep_reward)
        lengths.append(ep_length)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }


if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("Random Baseline Results")
    print("=" * 60)

    for env_id in cfg.env_ids:
        stats = random_baseline(
            env_id,
            num_episodes=100,
            seed=cfg.seed,
            max_steps=cfg.max_episode_steps,
        )
        print(f"\n  {env_id}")
        print(f"    reward:  {stats['mean_reward']:+.3f} ± {stats['std_reward']:.3f}")
        print(f"    length:  {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")

    print("\n" + "=" * 60)
    print("These are the floors to beat. Any learned policy should")
    print("significantly exceed these numbers.")
    print("=" * 60)
