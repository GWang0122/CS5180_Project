"""Temporal retention probe for MemoryS11 belief states.

Loads a trained (or randomly initialized) WorldModelAgent checkpoint, rolls it
out in MiniGrid-MemoryS11-v0, and records (h_t, cue_label, step_in_episode)
tuples. A per-bin logistic regression probe is then trained on h_t to predict
the identity of the cue object (Key vs Ball) shown in the starting room, and
probe accuracy is reported as a function of step-within-episode bin.

Why this matters: MemoryS11 evaluation return can sit at ~0.5 either because
the belief state *forgot* the cue (then the decision at the junction is random)
or because the belief state *remembered* the cue but the policy head failed
to condition on it. The probe separates these two hypotheses.

Usage (PowerShell):
    python analysis\temporal_retention_probe.py \
        --checkpoint runs\world_model_MiniGrid-MemoryS11-v0_s0.pt \
        --hidden-dim 256 --num-episodes 400 --seed 123 \
        --out results\MemoryS11\probe_s0_trained.csv

    python analysis\temporal_retention_probe.py --random-init \
        --hidden-dim 256 --num-episodes 400 --seed 123 \
        --out results\MemoryS11\probe_random_baseline.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from minigrid.core.world_object import Key, Ball

from common.env_wrappers import FlattenImageObs
from world_model.model import WorldModelAgent


# ----------------------------------------------------------------------
# Cue label extraction
# ----------------------------------------------------------------------
# MemoryS11 places a Key XOR a Ball in the starting room at grid position
# (1, 4). The agent must remember which one it saw and then pick the
# matching object at the far junction. The cue label is therefore:
#   0 = Ball, 1 = Key.

CUE_POS = (1, 4)


def extract_cue_label(unwrapped_env) -> int:
    obj = unwrapped_env.grid.get(*CUE_POS)
    if isinstance(obj, Key):
        return 1
    if isinstance(obj, Ball):
        return 0
    raise RuntimeError(
        f"No Key/Ball at {CUE_POS}; got {type(obj).__name__}. "
        "Environment may not be MemoryS11 or cue placement changed."
    )


# ----------------------------------------------------------------------
# Rollout data collection
# ----------------------------------------------------------------------

def collect_hidden_trajectories(
    model: WorldModelAgent,
    env_id: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
):
    """Roll out `num_episodes` episodes, return arrays of per-step (h_t, label, step)."""
    env = gym.make(env_id, max_steps=max_steps)
    env = FlattenImageObs(env)

    hs, labels, steps, rewards_per_ep = [], [], [], []
    model.eval()

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            cue = extract_cue_label(env.unwrapped)
            state = model.init_hidden(batch_size=1, device=device)
            hidden, _ = state
            t = 0
            ep_rew = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # single-step through encoder
                hidden_prev, prev_a = state
                hidden = model.encode_step(obs_t, prev_a, hidden_prev)
                dist, _ = model.get_policy_value(hidden)
                action = dist.sample()

                hs.append(hidden.squeeze(0).cpu().numpy())
                labels.append(cue)
                steps.append(t)

                obs, r, term, trunc, _ = env.step(action.item())
                ep_rew += float(r)
                done = term or trunc
                state = (hidden, action)
                t += 1

            rewards_per_ep.append(ep_rew)

    env.close()
    hs = np.stack(hs, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    rewards_per_ep = np.asarray(rewards_per_ep, dtype=np.float32)
    return hs, labels, steps, rewards_per_ep


# ----------------------------------------------------------------------
# Logistic regression probe (implemented in torch to avoid sklearn dep)
# ----------------------------------------------------------------------

def train_probe(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Fit a 2-class logistic regression on h -> y. Returns (train_acc, test_acc)."""
    h_tr = torch.as_tensor(h_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    h_te = torch.as_tensor(h_test, dtype=torch.float32, device=device)
    y_te = torch.as_tensor(y_test, dtype=torch.long, device=device)

    D = h_tr.shape[1]
    clf = nn.Linear(D, 2).to(device)
    nn.init.zeros_(clf.weight)
    nn.init.zeros_(clf.bias)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(clf(h_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (clf(h_tr).argmax(-1) == y_tr).float().mean().item()
        te_acc = (clf(h_te).argmax(-1) == y_te).float().mean().item()
    return tr_acc, te_acc


# ----------------------------------------------------------------------
# Bin scheme
# ----------------------------------------------------------------------
# Step-within-episode bins, chosen to span the natural phases of a
# MemoryS11 episode (agent starts in the cue room, walks a corridor,
# arrives at the junction). Boundaries are inclusive on the low end.

DEFAULT_BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 10_000)]
BIN_NAMES = ["cue-room(0-4)", "corridor-early(5-14)", "corridor-mid(15-29)",
             "junction(30-59)", "late(60+)"]


def split_by_bin(h: np.ndarray, y: np.ndarray, steps: np.ndarray,
                 bins=DEFAULT_BINS):
    out = []
    for lo, hi in bins:
        mask = (steps >= lo) & (steps < hi)
        out.append((h[mask], y[mask]))
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pt state dict. Omit when using --random-init.")
    ap.add_argument("--random-init", action="store_true",
                    help="Skip checkpoint loading; use freshly initialized weights.")
    ap.add_argument("--env", type=str, default="MiniGrid-MemoryS11-v0")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-episodes", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, required=True,
                    help="Output CSV path (per-bin accuracy).")
    args = ap.parse_args()

    gym.register_envs(minigrid)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Build a scratch env to read act_dim / obs_dim
    probe_env = gym.make(args.env, max_steps=args.max_steps)
    probe_env = FlattenImageObs(probe_env)
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.n
    probe_env.close()

    model = WorldModelAgent(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    if args.random_init:
        tag = "random-init"
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint required unless --random-init")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        tag = f"ckpt={os.path.basename(args.checkpoint)}"

    print(f"[probe] {tag} | env={args.env} | hidden_dim={args.hidden_dim} | "
          f"episodes={args.num_episodes} | device={device}")

    hs, labels, steps, ep_rewards = collect_hidden_trajectories(
        model, args.env, args.num_episodes, args.max_steps,
        args.seed, device,
    )
    print(f"[probe] collected {len(hs)} steps from {args.num_episodes} episodes | "
          f"mean reward = {ep_rewards.mean():.3f} ± {ep_rewards.std():.3f}")

    # Class balance over *episodes* (one label per episode)
    ep_labels = []
    t_idx = 0
    for ep in range(args.num_episodes):
        ep_labels.append(labels[t_idx])
        # advance t_idx to first step of next episode
        # (episodes are contiguous because we append step-by-step)
        # but this isn't strictly needed beyond the check below
        ep_len = int((steps[t_idx:] == 0).argmax() + 1) if ep < args.num_episodes - 1 else len(steps) - t_idx
        t_idx += ep_len
    ep_labels = np.asarray(ep_labels)
    print(f"[probe] cue balance across episodes: "
          f"Key={int((ep_labels==1).sum())} Ball={int((ep_labels==0).sum())}")

    # Shuffle episodes, then split 70/30 at the episode level to avoid
    # train/test contamination from adjacent timesteps.
    rng = np.random.default_rng(args.seed)
    # Rebuild an episode index aligned with (h, y, step)
    ep_id = np.zeros(len(hs), dtype=np.int64)
    cur = 0
    for i in range(1, len(hs)):
        if steps[i] == 0:
            cur += 1
        ep_id[i] = cur
    n_ep = int(ep_id.max()) + 1
    perm = rng.permutation(n_ep)
    n_train = int(args.train_frac * n_ep)
    train_eps = set(perm[:n_train].tolist())
    train_mask = np.isin(ep_id, list(train_eps))
    test_mask = ~train_mask

    rows = []
    for (lo, hi), name in zip(DEFAULT_BINS, BIN_NAMES):
        bin_mask = (steps >= lo) & (steps < hi)
        tr_m = bin_mask & train_mask
        te_m = bin_mask & test_mask
        n_tr, n_te = int(tr_m.sum()), int(te_m.sum())
        if n_tr < 20 or n_te < 20:
            print(f"  bin {name:>22s}: n_train={n_tr} n_test={n_te}  "
                  f"(skipped, too few samples)")
            rows.append({"bin": name, "step_lo": lo, "step_hi": hi,
                         "n_train": n_tr, "n_test": n_te,
                         "train_acc": None, "test_acc": None,
                         "majority_baseline": None})
            continue
        # Majority-class baseline (per-bin)
        maj = float(max((labels[te_m] == 0).mean(), (labels[te_m] == 1).mean()))
        tr_acc, te_acc = train_probe(
            hs[tr_m], labels[tr_m], hs[te_m], labels[te_m], device=device,
        )
        print(f"  bin {name:>22s}: n_train={n_tr:>5d} n_test={n_te:>5d}  "
              f"train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}  "
              f"majority={maj:.3f}")
        rows.append({"bin": name, "step_lo": lo, "step_hi": hi,
                     "n_train": n_tr, "n_test": n_te,
                     "train_acc": tr_acc, "test_acc": te_acc,
                     "majority_baseline": maj})

    # Write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[probe] wrote {args.out}")
    print(f"[probe] source tag: {tag}")


if __name__ == "__main__":
    main()
