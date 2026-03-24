"""Deterministic world model agent inspired by PlaNet/Dreamer.

Architecture:
    Belief state:           h_t = GRU(h_{t-1}, [enc(o_t); emb(a_{t-1})])
    Transition model:       ĥ_{t+1} = f_φ(h_t, a_t)
    Observation predictor:  ô_{t+1} = g_ψ(h_t, a_t)
    Policy & value:         π(a_t | h_t),  V(h_t)

The GRU accumulates history into a belief state h_t.  Two world-model heads
provide auxiliary supervision:
  - The transition model predicts the next hidden state.
  - The observation predictor decodes the expected next observation from
    (h_t, a_t), grounding representations in concrete environment details.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho_init(module, gain=np.sqrt(2)):
    """Apply orthogonal initialization (standard for PPO)."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class WorldModelAgent(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128,
                 obs_embed_dim: int = 64, act_embed_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # CNN matching the reference MiniGrid architecture (rl-starter-files).
        # MaxPool after first conv compresses 7x7 → 3x3, keeping the final
        # feature count at 64 instead of 1024.
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),                                  # 64 * 1 * 1 = 64
            nn.Linear(64, obs_embed_dim),
            nn.ReLU(),
        )
        self.act_embed = nn.Embedding(act_dim, act_embed_dim)

        self.gru = nn.GRUCell(obs_embed_dim + act_embed_dim, hidden_dim)

        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + act_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.obs_predictor = nn.Sequential(
            nn.Linear(hidden_dim + act_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Feature extraction layers: gain = sqrt(2) for ReLU
        for module in self.obs_encoder:
            _ortho_init(module, gain=np.sqrt(2))
        for module in self.transition:
            _ortho_init(module, gain=np.sqrt(2))
        for module in self.obs_predictor:
            _ortho_init(module, gain=np.sqrt(2))

        # Actor: small gain on the final layer → near-uniform initial policy
        _ortho_init(self.actor[0], gain=np.sqrt(2))  # hidden layer
        _ortho_init(self.actor[2], gain=0.01)         # output logits

        # Critic: gain 1.0 on the final layer → moderate initial values
        _ortho_init(self.critic[0], gain=np.sqrt(2))  # hidden layer
        _ortho_init(self.critic[2], gain=1.0)          # output value

        # GRU: orthogonal init on all weight matrices
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Core forward components
    # ------------------------------------------------------------------

    def _encode_obs(self, obs_flat):
        """Reshape flat obs to (B, 3, 7, 7) and run through CNN encoder."""
        B = obs_flat.shape[0]
        img = obs_flat.view(B, 7, 7, 3).permute(0, 3, 1, 2)
        return self.obs_encoder(img)

    def encode_step(self, obs, prev_action, hidden):
        """Single GRU step:  h_t = GRU(h_{t-1}, [enc(o_t); emb(a_{t-1})]).

        Args:
            obs:         (B, obs_dim)  flat 147-dim vector
            prev_action: (B,) long
            hidden:      (B, hidden_dim)
        Returns:
            (B, hidden_dim)
        """
        obs_enc = self._encode_obs(obs)
        act_enc = self.act_embed(prev_action)
        return self.gru(torch.cat([obs_enc, act_enc], dim=-1), hidden)

    def get_policy_value(self, hidden):
        """Policy distribution and value from belief state.

        Args:
            hidden: (B, hidden_dim)
        Returns:
            dist (Categorical), value (B,)
        """
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return Categorical(logits=logits), value

    def predict_next_hidden(self, hidden, action):
        """Transition model:  ĥ_{t+1} = f_φ(h_t, a_t).

        Args:
            hidden: (B, hidden_dim)
            action: (B,) long
        Returns:
            (B, hidden_dim)
        """
        act_enc = self.act_embed(action)
        return self.transition(torch.cat([hidden, act_enc], dim=-1))

    def predict_next_obs(self, hidden, action):
        """Observation predictor:  ô_{t+1} = g_ψ(h_t, a_t).

        Args:
            hidden: (B, hidden_dim)
            action: (B,) long
        Returns:
            (B, obs_dim)
        """
        act_enc = self.act_embed(action)
        return self.obs_predictor(torch.cat([hidden, act_enc], dim=-1))

    # ------------------------------------------------------------------
    # Interface expected by common.evaluation.evaluate()
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu")):
        """Returns (gru_hidden, prev_action) tuple."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        prev_a = torch.zeros(batch_size, dtype=torch.long, device=device)
        return (h, prev_a)

    @torch.no_grad()
    def step(self, obs, state):
        """Single-step for rollout collection and evaluation.

        Args:
            obs:   (1, obs_dim) tensor
            state: (hidden, prev_action) from init_hidden or prior step
        Returns:
            action (int), log_prob (float), value (float), new_state
        """
        hidden, prev_action = state
        hidden = self.encode_step(obs, prev_action, hidden)
        dist, value = self.get_policy_value(hidden)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), (hidden, action)
