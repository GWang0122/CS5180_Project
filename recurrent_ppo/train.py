"""Training loop for recurrent PPO with parallel MiniGrid environments.

Usage:
    python -m recurrent_ppo.train --env MiniGrid-DoorKey-8x8-v0
    python -m recurrent_ppo.train --env MiniGrid-MemoryS11-v0 --seed 123
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
from recurrent_ppo.model import RecurrentPPOAgent


def train(cfg: Config, env_id: str, seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_envs = cfg.num_envs
    rollout_len = cfg.rollout_length

    envs = make_vec_env(env_id, num_envs, seed=seed, max_steps=cfg.max_episode_steps)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = RecurrentPPOAgent(obs_dim, act_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    logger = Logger(cfg.log_dir, "recurrent_ppo", env_id, seed)

    obs, _ = envs.reset()
    hidden = torch.zeros(num_envs, cfg.hidden_dim, device=device)
    prev_action = torch.zeros(num_envs, dtype=torch.long, device=device)

    ep_rewards = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=int)
    global_step = 0

    num_updates = cfg.total_timesteps // (rollout_len * num_envs)

    for update in range(num_updates):
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        buf_obs = torch.zeros(rollout_len, num_envs, obs_dim, device=device)
        buf_prev_act = torch.zeros(rollout_len, num_envs, dtype=torch.long, device=device)
        buf_act = torch.zeros(rollout_len, num_envs, dtype=torch.long, device=device)
        buf_logprob = torch.zeros(rollout_len, num_envs, device=device)
        buf_value = torch.zeros(rollout_len, num_envs, device=device)
        buf_reward = torch.zeros(rollout_len, num_envs, device=device)
        buf_done = torch.zeros(rollout_len, num_envs, device=device)

        init_hidden = hidden.detach().clone()

        model.eval()
        with torch.no_grad():
            for t in range(rollout_len):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

                hidden = model.encode_step(obs_t, prev_action, hidden)
                dist, value = model.get_policy_value(hidden)
                action = dist.sample()

                buf_obs[t] = obs_t
                buf_prev_act[t] = prev_action
                buf_act[t] = action
                buf_logprob[t] = dist.log_prob(action)
                buf_value[t] = value

                obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                done = terminated | truncated

                buf_reward[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                buf_done[t] = torch.as_tensor(done, dtype=torch.float32, device=device)

                for i in range(num_envs):
                    ep_rewards[i] += reward[i]
                    ep_lengths[i] += 1
                    if done[i]:
                        logger.log_episode(
                            ep_rewards[i], int(ep_lengths[i]), global_step + num_envs
                        )
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0

                done_t = buf_done[t]
                hidden = hidden * (1.0 - done_t.unsqueeze(-1))
                prev_action = action * (1 - done_t.long())

                global_step += num_envs

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            h_boot = model.encode_step(obs_t, prev_action, hidden)
            _, bootstrap_values = model.get_policy_value(h_boot)

        advantages = torch.zeros(rollout_len, num_envs, device=device)
        returns = torch.zeros(rollout_len, num_envs, device=device)

        for i in range(num_envs):
            advantages[:, i], returns[:, i] = compute_gae(
                buf_reward[:, i],
                buf_value[:, i],
                buf_done[:, i],
                bootstrap_values[i],
                cfg.gamma,
                cfg.gae_lambda,
            )

        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(rollout_len, num_envs)

        model.train()
        tbptt = cfg.tbptt_len
        num_minibatches = max(1, min(cfg.num_minibatches, num_envs))
        envs_per_minibatch = num_envs // num_minibatches

        for _ in range(cfg.ppo_epochs):
            env_perm = torch.randperm(num_envs, device=device)

            for mb in range(num_minibatches):
                start = mb * envs_per_minibatch
                end = num_envs if mb == num_minibatches - 1 else (mb + 1) * envs_per_minibatch
                mb_idx = env_perm[start:end]

                h = init_hidden[mb_idx].clone()
                hiddens = []

                for t in range(rollout_len):
                    if t > 0:
                        h = h * (1.0 - buf_done[t - 1, mb_idx].unsqueeze(-1))
                    if t % tbptt == 0 and t > 0:
                        h = h.detach()
                    h = model.encode_step(buf_obs[t, mb_idx], buf_prev_act[t, mb_idx], h)
                    hiddens.append(h)

                hiddens = torch.stack(hiddens)

                tn = rollout_len * mb_idx.numel()
                h_flat = hiddens.reshape(tn, -1)
                act_flat = buf_act[:, mb_idx].reshape(tn)
                old_lp_flat = buf_logprob[:, mb_idx].reshape(tn)
                adv_flat = advantages[:, mb_idx].reshape(tn)
                ret_flat = returns[:, mb_idx].reshape(tn)

                dist, new_values = model.get_policy_value(h_flat)
                new_lp = dist.log_prob(act_flat)
                entropy = dist.entropy()

                ratio = (new_lp - old_lp_flat).exp()
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_flat

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_values - ret_flat).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        logger.log_losses(policy_loss.item(), value_loss.item(), entropy.mean().item(), global_step)

        if global_step % cfg.eval_interval < (rollout_len * num_envs):
            stats = evaluate(
                model,
                env_id,
                cfg.eval_episodes,
                seed=seed + 10_000,
                max_steps=cfg.max_episode_steps,
                device=device,
            )
            logger.log_scalar("eval/mean_reward", stats["mean_reward"], global_step)
            logger.log_scalar("eval/mean_length", stats["mean_length"], global_step)
            print(
                f"  [EVAL] step {global_step:>7d} | "
                f"R={stats['mean_reward']:+.3f} +- {stats['std_reward']:.3f} | "
                f"len={stats['mean_length']:.1f}"
            )

    os.makedirs(cfg.log_dir, exist_ok=True)
    save_path = os.path.join(cfg.log_dir, f"recurrent_ppo_{env_id}_s{seed}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    logger.close()
    envs.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
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
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Recurrent PPO | {args.env} | seed={args.seed} | device={device}")
    print(f"  timesteps={cfg.total_timesteps}  num_envs={cfg.num_envs}  rollout={cfg.rollout_length}")
    print(f"  ent_coef={cfg.ent_coef}  lr={cfg.lr}")

    train(cfg, args.env, args.seed, device)
