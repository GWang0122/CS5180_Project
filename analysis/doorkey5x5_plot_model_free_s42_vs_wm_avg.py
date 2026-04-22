"""Plot DoorKey-5x5: model-free (seed 42 only) vs world-model (avg seeds 0, 42)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MF_S42 = ROOT / "results" / "Doorkey5x5" / "model_free" / "model_free_5x5_s42.csv"
WM_AVG = (
    ROOT
    / "results"
    / "Doorkey5x5"
    / "run-world_model_MiniGrid-DoorKey-5x5-v0_avg_s0_s42-tag-episode_reward.csv"
)
FIG_OUT = ROOT / "results" / "Doorkey5x5" / "doorkey5x5_model_free_s42_vs_wm_avg.png"


def load_step_value(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        keys = {k.lower().strip(): k for k in (r.fieldnames or [])}
        sk = keys.get("step")
        vk = keys.get("value")
        if not sk or not vk:
            raise ValueError(f"Missing step/value columns in {path}: {r.fieldnames}")
        for row in r:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            st = int(float(str(row[sk]).strip()))
            val = float(str(row[vk]).strip())
            rows.append((st, val))
    return rows


def main() -> None:
    mf = load_step_value(MF_S42)
    wm = load_step_value(WM_AVG)

    plt.figure(figsize=(8, 4.5))
    plt.plot([s for s, _ in mf], [v for _, v in mf], label="Model-free (seed 42)", linewidth=1.8)
    plt.plot(
        [s for s, _ in wm],
        [v for _, v in wm],
        label="World model (avg seeds 0, 42)",
        linewidth=1.8,
        alpha=0.9,
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward (logged)")
    plt.title("DoorKey-5x5: model-free (s42) vs world model (mean s0, s42)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=150)
    plt.close()
    print(f"Wrote {FIG_OUT} (model-free points={len(mf)}, world-model points={len(wm)})")


if __name__ == "__main__":
    main()
