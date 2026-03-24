"""Shared hyperparameters and configuration for all algorithms."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Environment ---
    env_ids: list[str] = field(
        default_factory=lambda: [
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-MemoryS11-v0",
        ]
    )
    max_episode_steps: int = 256

    # --- Training ---
    total_timesteps: int = 5_000_000
    rollout_length: int = 256
    num_envs: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42

    # --- PPO ---
    ppo_epochs: int = 4
    num_minibatches: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.05
    max_grad_norm: float = 0.5
    lr: float = 7e-4

    # --- Network ---
    hidden_dim: int = 128
    tbptt_len: int = 16

    # --- Evaluation ---
    eval_interval: int = 10_000
    eval_episodes: int = 30

    # --- Logging ---
    log_dir: str = "runs"

    @property
    def batch_size(self) -> int:
        return self.rollout_length * self.num_envs

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches
