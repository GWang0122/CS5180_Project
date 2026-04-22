"""DoorKey8x8: average eval_mean_reward over seeds for model-free and world model,
then plot both curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MF_S0 = ROOT / "results" / "Doorkey8x8" / "model_free" / "doorkey8x8_s0.csv"
MF_S42 = ROOT / "results" / "Doorkey8x8" / "model_free" / "doorkey8x8_s42.csv"
WM_S0 = (
    ROOT
    / "results"
    / "Doorkey8x8"
    / "world_model"
    / "run-world_model_MiniGrid-DoorKey-8x8-v0_s0_1776558051-tag-eval_mean_reward.csv"
)
WM_S42 = (
    ROOT
    / "results"
    / "Doorkey8x8"
    / "world_model"
    / "run-world_model_MiniGrid-DoorKey-8x8-v0_s42_1776481078-tag-eval_mean_reward.csv"
)

MF_OUT = ROOT / "results" / "Doorkey8x8" / "model_free" / "doorkey8x8_avg_s0_s42.csv"
WM_OUT = ROOT / "results" / "Doorkey8x8" / "world_model" / "doorkey8x8_avg_s0_s42.csv"
FIG_OUT = ROOT / "results" / "Doorkey8x8" / "doorkey8x8_model_free_avg_vs_world_model_avg.png"


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


def average_two(
    a: list[tuple[float, int, float]],
    b: list[tuple[float, int, float]],
    label: str,
) -> list[tuple[float, int, float]]:
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError(f"{label}: empty")
    for i in range(n):
        if a[i][1] != b[i][1]:
            raise ValueError(
                f"{label}: step mismatch at row {i}: {a[i][1]} vs {b[i][1]}"
            )
    out: list[tuple[float, int, float]] = []
    for i in range(n):
        wt = (a[i][0] + b[i][0]) / 2.0
        st = a[i][1]
        val = (a[i][2] + b[i][2]) / 2.0
        out.append((wt, st, val))
    return out


def write_csv(path: Path, rows: list[tuple[float, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wall_time", "step", "value"])
        for wt, st, val in rows:
            w.writerow([wt, st, val])


def main() -> None:
    mf_avg = average_two(load_rows(MF_S0), load_rows(MF_S42), "model-free")
    wm_avg = average_two(load_rows(WM_S0), load_rows(WM_S42), "world-model")

    write_csv(MF_OUT, mf_avg)
    write_csv(WM_OUT, wm_avg)
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
        label="World model (mean seeds 0, 42)",
        linewidth=1.8,
        alpha=0.9,
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Eval mean reward")
    plt.title("DoorKey8x8: model-free vs world model (seed-mean curves)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=150)
    plt.close()
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()

