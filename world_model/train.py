"""Training loop for the World Model Agent with parallel environments.

Collects rollouts from N parallel envs, then jointly optimises:
  1. PPO clipped-surrogate policy loss
  2. Value-function loss
  3. Transition-model prediction loss  (ĥ_{t+1} ≈ h_{t+1})
  4. Observation prediction loss        (ô_{t+1} ≈ o_{t+1})

Usage:
    python -m world_model.train --env MiniGrid-DoorKey-8x8-v0
    python -m world_model.train --env MiniGrid-MemoryS11-v0 --seed 123
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from common.env_wrappers import make_vec_env
from common.logger import Logger
from common.evaluation import evaluate
from common.rollout_buffer import compute_gae
from world_model.model import WorldModelAgent


def train(cfg: Config, env_id: str, seed: int, device: torch.device,
          transition_coef: float = 0.5, obs_pred_coef: float = 0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = cfg.num_envs
    T = cfg.rollout_length

    envs = make_vec_env(env_id, N, seed=seed, max_steps=cfg.max_episode_steps)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = WorldModelAgent(obs_dim, act_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    logger = Logger(cfg.log_dir, "world_model", env_id, seed)

    # ---- persistent state across rollouts ----
    obs, _ = envs.reset()                                        # (N, obs_dim)
    hidden = torch.zeros(N, cfg.hidden_dim, device=device)       # (N, H)
    prev_action = torch.zeros(N, dtype=torch.long, device=device)
    ep_rewards = np.zeros(N)
    ep_lengths = np.zeros(N, dtype=int)
    global_step = 0

    num_updates = cfg.total_timesteps // (T * N)

    for update in range(num_updates):
        # Linear LR annealing
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        # ==============================================================
        # 1.  Collect rollout  (T steps × N envs)
        # ==============================================================
        buf_obs = torch.zeros(T, N, obs_dim, device=device)
        buf_prev_act = torch.zeros(T, N, dtype=torch.long, device=device)
        buf_act = torch.zeros(T, N, dtype=torch.long, device=device)
        buf_logprob = torch.zeros(T, N, device=device)
        buf_value = torch.zeros(T, N, device=device)
        buf_reward = torch.zeros(T, N, device=device)
        buf_done = torch.zeros(T, N, device=device)

        init_hidden = hidden.detach().clone()

        model.eval()
        with torch.no_grad():
            for t in range(T):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

                hidden = model.encode_step(obs_t, prev_action, hidden)
                dist, value = model.get_policy_value(hidden)
                action = dist.sample()

                buf_obs[t] = obs_t
                buf_prev_act[t] = prev_action
                buf_act[t] = action
                buf_logprob[t] = dist.log_prob(action)
                buf_value[t] = value

                obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                done = terminated | truncated

                buf_reward[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                buf_done[t] = torch.as_tensor(done, dtype=torch.float32, device=device)

                for i in range(N):
                    ep_rewards[i] += reward[i]
                    ep_lengths[i] += 1
                    if done[i]:
                        logger.log_episode(ep_rewards[i], int(ep_lengths[i]),
                                           global_step)
                        ep_rewards[i] = 0
                        ep_lengths[i] = 0

                done_t = buf_done[t]
                hidden = hidden * (1.0 - done_t.unsqueeze(-1))
                prev_action = action * (1 - done_t.long())

                global_step += N

            # Bootstrap V(s_{T+1})
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            h_boot = model.encode_step(obs_t, prev_action, hidden)
            _, v_boot = model.get_policy_value(h_boot)
            bootstrap_values = v_boot                              # (N,)

        # ==============================================================
        # 2.  Compute GAE per environment
        # ==============================================================
        advantages = torch.zeros(T, N, device=device)
        returns = torch.zeros(T, N, device=device)
        for i in range(N):
            advantages[:, i], returns[:, i] = compute_gae(
                buf_reward[:, i], buf_value[:, i], buf_done[:, i],
                bootstrap_values[i], cfg.gamma, cfg.gae_lambda,
            )
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(T, N)

        # ==============================================================
        # 3.  PPO + world-model update
        # ==============================================================
        model.train()
        tbptt = cfg.tbptt_len
        for _epoch in range(cfg.ppo_epochs):
            # -- replay sequences with truncated BPTT --
            h = init_hidden.clone()
            hiddens = []
            for t in range(T):
                if t > 0:
                    h = h * (1.0 - buf_done[t - 1].unsqueeze(-1))
                if t % tbptt == 0 and t > 0:
                    h = h.detach()
                h = model.encode_step(buf_obs[t], buf_prev_act[t], h)
                hiddens.append(h)
            hiddens = torch.stack(hiddens)                        # (T, N, H)

            # -- flatten (T, N) → (T*N) for loss computation --
            TN = T * N
            h_flat = hiddens.reshape(TN, -1)
            act_flat = buf_act.reshape(TN)
            old_lp_flat = buf_logprob.reshape(TN)
            adv_flat = advantages.reshape(TN)
            ret_flat = returns.reshape(TN)

            dist, new_values = model.get_policy_value(h_flat)
            new_lp = dist.log_prob(act_flat)
            entropy = dist.entropy()

            # PPO clipped surrogate
            ratio = (new_lp - old_lp_flat).exp()
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps,
                                1 + cfg.clip_eps) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (new_values - ret_flat).pow(2).mean()
            entropy_loss = -entropy.mean()

            # -- world-model losses across valid (non-episode-boundary) pairs --
            if T > 1:
                h_src = hiddens[:-1].reshape((T - 1) * N, -1)
                a_src = buf_act[:-1].reshape((T - 1) * N)
                valid = (1.0 - buf_done[:-1]).reshape((T - 1) * N)
                n_valid = valid.sum() + 1e-8

                pred_h = model.predict_next_hidden(h_src, a_src)
                target_h = hiddens[1:].reshape((T - 1) * N, -1).detach()
                trans_loss = ((pred_h - target_h).pow(2).sum(-1) * valid).sum() / n_valid

                pred_obs = model.predict_next_obs(h_src, a_src)
                target_obs = buf_obs[1:].reshape((T - 1) * N, -1)
                obs_pred_loss = ((pred_obs - target_obs).pow(2).sum(-1) * valid).sum() / n_valid
            else:
                trans_loss = torch.tensor(0.0, device=device)
                obs_pred_loss = torch.tensor(0.0, device=device)

            loss = (
                policy_loss
                + cfg.vf_coef * value_loss
                + cfg.ent_coef * entropy_loss
                + transition_coef * trans_loss
                + obs_pred_coef * obs_pred_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # ==============================================================
        # 4.  Logging & periodic evaluation
        # ==============================================================
        logger.log_losses(
            policy_loss.item(), value_loss.item(), entropy.mean().item(),
            global_step,
        )
        logger.log_scalar("loss/transition", trans_loss.item(), global_step)
        logger.log_scalar("loss/obs_prediction", obs_pred_loss.item(), global_step)

        if global_step % cfg.eval_interval < (T * N):
            stats = evaluate(
                model, env_id, cfg.eval_episodes,
                seed=seed + 10_000, max_steps=cfg.max_episode_steps,
                device=device,
            )
            logger.log_scalar("eval/mean_reward", stats["mean_reward"], global_step)
            logger.log_scalar("eval/mean_length", stats["mean_length"], global_step)
            print(
                f"  [EVAL] step {global_step:>7d} | "
                f"R={stats['mean_reward']:+.3f} ± {stats['std_reward']:.3f} | "
                f"len={stats['mean_length']:.1f}"
            )

    # ---- save & cleanup ----
    os.makedirs(cfg.log_dir, exist_ok=True)
    save_path = os.path.join(cfg.log_dir, f"world_model_{env_id}_s{seed}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    logger.close()
    envs.close()
    return model


# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--transition-coef", type=float, default=0.1)
    parser.add_argument("--obs-pred-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.num_envs:
        cfg.num_envs = args.num_envs
    if args.ent_coef is not None:
        cfg.ent_coef = args.ent_coef
    if args.lr is not None:
        cfg.lr = args.lr

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"World Model Agent | {args.env} | seed={args.seed} | device={device}")
    print(f"  timesteps={cfg.total_timesteps}  num_envs={cfg.num_envs}  "
          f"rollout={cfg.rollout_length}")
    print(f"  transition_coef={args.transition_coef}  "
          f"obs_pred_coef={args.obs_pred_coef}  ent_coef={cfg.ent_coef}")
    train(cfg, args.env, args.seed, device,
          transition_coef=args.transition_coef,
          obs_pred_coef=args.obs_pred_coef)
