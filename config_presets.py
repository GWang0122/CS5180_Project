"""Named hyperparameter bundles for `world_model.train`.

Use `--preset doorkey`, `--preset doorkey5x5`, or `--preset memory` to load task-specific defaults.
`--env`, `--num-envs`, `--curiosity-coef`, etc. still override the preset if passed.
"""

from __future__ import annotations

from typing import Any

from config import Config

# Each preset: default env id + Config fields to set (must match Config attribute names).
PRESETS: dict[str, dict[str, Any]] = {
    "doorkey": {
        "env": "MiniGrid-DoorKey-8x8-v0",
        "curiosity_coef": 0.0,
        "num_envs": 48,
        "imagine_horizon": 0,
        "transition_coef": 0.25,
        "obs_pred_coef": 0.25,
        "ent_coef": 0.025,
        "hidden_dim": 256,
        "tbptt_len": 32,
    },
    # Small-grid sanity preset (matches earlier project defaults that solved 5x5 reliably).
    "doorkey5x5": {
        "env": "MiniGrid-DoorKey-5x5-v0",
        "curiosity_coef": 0.0,
        "num_envs": 16,
        "imagine_horizon": 0,
        "transition_coef": 0.1,
        "obs_pred_coef": 0.1,
        "ent_coef": 0.05,
        "hidden_dim": 128,
        "tbptt_len": 16,
        "max_episode_steps": 256,
        "lr": 7e-4,
    },
    "memory": {
        "env": "MiniGrid-MemoryS11-v0",
        "curiosity_coef": 0.005,
        "num_envs": 48,
        "imagine_horizon": 0,
        "transition_coef": 0.25,
        "obs_pred_coef": 0.25,
        "ent_coef": 0.08,
        "hidden_dim": 256,
        "tbptt_len": 64,
        "max_episode_steps": 256,
        "eval_episodes": 100,
    },
}


def apply_preset(cfg: Config, name: str) -> str:
    """Apply preset fields to cfg. Returns the default env id for this preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {list(PRESETS)}")
    spec = PRESETS[name]
    env_id = spec["env"]
    for key, value in spec.items():
        if key == "env":
            continue
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field {key!r} (preset {name!r})")
        setattr(cfg, key, value)
    return env_id
