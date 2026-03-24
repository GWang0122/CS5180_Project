"""Shared environment creation and wrappers for MiniGrid experiments."""

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import ObservationWrapper, spaces

gym.register_envs(minigrid)


class FlattenImageObs(ObservationWrapper):
    """Extract the 'image' key from MiniGrid's Dict obs and flatten to 1-D float32.

    MiniGrid observations are Dict with 'image' (7,7,3), 'direction', and 'mission'.
    We keep only the partial-view image grid and flatten it to a 147-dim vector.

    MiniGrid image channels encode small categorical integers, NOT pixel values:
        ch0 = object type (0-10), ch1 = color (0-5), ch2 = state (0-3).
    We normalize each channel by its respective max so values lie in [0, 1].
    """

    _CHANNEL_MAX = np.array([10.0, 5.0, 3.0], dtype=np.float32)

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        self._flat_dim = int(np.prod(img_space.shape))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"].astype(np.float32)
        img /= self._CHANNEL_MAX
        return img.flatten()


def make_env(env_id: str, seed: int = 0, max_steps: int = 256, render_mode=None):
    """Create a single MiniGrid environment with standard wrappers."""
    env = gym.make(env_id, max_steps=max_steps, render_mode=render_mode)
    env = FlattenImageObs(env)
    env.reset(seed=seed)
    return env


def make_vec_env(env_id: str, num_envs: int, seed: int = 0,
                 max_steps: int = 256):
    """Create N parallel MiniGrid environments (SyncVectorEnv).

    Each sub-env gets a unique seed for layout diversity.
    Auto-resets on termination/truncation.
    """
    def _make(i):
        def _init():
            env = gym.make(env_id, max_steps=max_steps)
            env = FlattenImageObs(env)
            env.reset(seed=seed + i)
            return env
        return _init

    return gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
