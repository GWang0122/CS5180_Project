"""DoorKey-5x5: recompute seed-mean CSVs and plot model-free vs world model.

Model-free: mean of eval_mean_reward_s0.csv and model_free_5x5_s42.csv (aligned steps).
World model: mean of per-seed episode_reward CSVs; steps differ between seeds, so we
use seed-0's step grid and linearly interpolate seed-42 values onto those steps.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DK = ROOT / "results" / "Doorkey5x5"

MF_S0 = DK / "model_free" / "eval_mean_reward_s0.csv"
MF_S42 = DK / "model_free" / "model_free_5x5_s42.csv"
WM_S0 = DK / "world_model" / "world_model_MiniGrid-DoorKey-5x5-v0_s0_1776735063.csv"
WM_S42 = DK / "world_model" / "world_model_MiniGrid-DoorKey-5x5-v0_s42_1776731484.csv"

MF_OUT = DK / "model_free" / "eval_mean_reward_avg_s0_s42.csv"
WM_OUT = DK / "world_model" / "world_model_MiniGrid-DoorKey-5x5-v0_avg_s0_s42.csv"
FIG_OUT = DK / "doorkey5x5_seed_mean_mf_vs_wm.png"


def load_rows(path: Path) -> list[tuple[float, int, float]]:
    rows: list[tuple[float, int, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        keys = {k.lower().strip(): k for k in (r.fieldnames or [])}
        wt_k = keys.get("wall_time", keys.get("wall time"))
        sk = keys.get("step")
        vk = keys.get("value")
        if not all([wt_k, sk, vk]):
            raise ValueError(f"Bad headers in {path}: {r.fieldnames}")
        for row in r:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            wt = float(str(row[wt_k]).strip())
            st = int(float(str(row[sk]).strip()))
            val = float(str(row[vk]).strip())
            rows.append((wt, st, val))
    return rows


def average_aligned(
    a: list[tuple[float, int, float]],
    b: list[tuple[float, int, float]],
    name: str,
) -> list[tuple[float, int, float]]:
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError(f"{name}: empty")
    for i in range(n):
        if a[i][1] != b[i][1]:
            raise ValueError(f"{name}: step mismatch row {i}: {a[i][1]} vs {b[i][1]}")
    out = []
    for i in range(n):
        wt = (a[i][0] + b[i][0]) / 2.0
        st = a[i][1]
        val = (a[i][2] + b[i][2]) / 2.0
        out.append((wt, st, val))
    return out


def world_model_mean_on_s0_grid(
    rows0: list[tuple[float, int, float]],
    rows42: list[tuple[float, int, float]],
) -> list[tuple[float, int, float]]:
    """Interpolate seed-42 (wall, step, val) onto each step from seed-0."""
    s42 = np.array([r[1] for r in rows42], dtype=np.float64)
    v42 = np.array([r[2] for r in rows42], dtype=np.float64)
    t42 = np.array([r[0] for r in rows42], dtype=np.float64)
    order = np.argsort(s42)
    s42, v42, t42 = s42[order], v42[order], t42[order]
    # np.interp requires increasing x; duplicate steps would break—use unique last
    out_rows: list[tuple[float, int, float]] = []
    for wt0, st, v0 in rows0:
        v42i = float(np.interp(st, s42, v42))
        wt42i = float(np.interp(st, s42, t42))
        out_rows.append(((wt0 + wt42i) / 2.0, st, (v0 + v42i) / 2.0))
    return out_rows


def write_csv(path: Path, rows: list[tuple[float, int, float]], header_wall: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_wall, "Step", "Value"])
        for wt, st, val in rows:
            w.writerow([wt, st, val])


def main() -> None:
    mf0 = load_rows(MF_S0)
    mf42 = load_rows(MF_S42)
    mf_avg = average_aligned(mf0, mf42, "model-free")

    w0 = load_rows(WM_S0)
    w42 = load_rows(WM_S42)
    wm_avg = world_model_mean_on_s0_grid(w0, w42)

    write_csv(MF_OUT, mf_avg, "wall_time")
    write_csv(WM_OUT, wm_avg, "Wall time")
    print(f"Wrote {MF_OUT} ({len(mf_avg)} rows)")
    print(f"Wrote {WM_OUT} ({len(wm_avg)} rows)")

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        [r[1] for r in mf_avg],
        [r[2] for r in mf_avg],
        label="Model-free (mean seeds 0, 42)",
        linewidth=1.8,
    )
    plt.plot(
        [r[1] for r in wm_avg],
        [r[2] for r in wm_avg],
        label="World model (mean seeds 0, 42; s42 interp. to s0 steps)",
        linewidth=1.8,
        alpha=0.9,
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward (logged)")
    plt.title("DoorKey-5x5: seed-mean model-free vs world model")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150)
    plt.close()
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
