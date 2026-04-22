"""Recurrent PPO agent for partially observable MiniGrid tasks.

Architecture:
    h_t = GRU(h_{t-1}, [enc(o_t); emb(a_{t-1})])
    pi(a_t | h_t), V(h_t)

The observation encoder mirrors the world-model CNN so both methods use
comparable representational capacity before the recurrent core.
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


class RecurrentPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128,
                 obs_embed_dim: int = 64, act_embed_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, obs_embed_dim),
            nn.ReLU(),
        )

        self.act_embed = nn.Embedding(act_dim, act_embed_dim)
        self.gru = nn.GRUCell(obs_embed_dim + act_embed_dim, hidden_dim)

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
        for module in self.obs_encoder:
            _ortho_init(module, gain=np.sqrt(2))

        _ortho_init(self.actor[0], gain=np.sqrt(2))
        _ortho_init(self.actor[2], gain=0.01)

        _ortho_init(self.critic[0], gain=np.sqrt(2))
        _ortho_init(self.critic[2], gain=1.0)

        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _encode_obs(self, obs_flat):
        """Reshape flattened MiniGrid obs and run the CNN encoder."""
        batch_size = obs_flat.shape[0]
        img = obs_flat.view(batch_size, 7, 7, 3).permute(0, 3, 1, 2)
        return self.obs_encoder(img)

    def encode_step(self, obs, prev_action, hidden):
        """Run one recurrent update step."""
        obs_enc = self._encode_obs(obs)
        act_enc = self.act_embed(prev_action)
        return self.gru(torch.cat([obs_enc, act_enc], dim=-1), hidden)

    def get_policy_value(self, hidden):
        """Return categorical policy distribution and value estimate."""
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return Categorical(logits=logits), value

    def init_hidden(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu")):
        """Return recurrent state tuple expected by shared evaluator."""
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        return hidden, prev_action

    @torch.no_grad()
    def step(self, obs, state, deterministic: bool = False):
        """Pick one action and update recurrent state.

        Args:
            obs: observation tensor with shape (B, obs_dim), typically B=1.
            state: tuple(hidden, prev_action).
            deterministic: if True, pick argmax action for evaluation.
        """
        hidden, prev_action = state
        hidden = self.encode_step(obs, prev_action, hidden)
        dist, value = self.get_policy_value(hidden)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), (hidden, action)
