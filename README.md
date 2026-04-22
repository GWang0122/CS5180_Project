# Model-Based vs. Model-Free RL in Partially Observable MiniGrid

CS 5180: Reinforcement Learning, Spring 2026 — final project.
George Wang, Shreyas Suresh Manoti.

We compare a **Recurrent PPO** baseline against a **deterministic world-model
agent** (GRU belief state + latent-transition head + observation-reconstruction
head + optional curiosity + optional Dyna-style imagination) on three
partially observable MiniGrid tasks: `DoorKey-5x5-v0`, `DoorKey-8x8-v0`, and
`MemoryS11-v0`. The full write-up (experiments, diagnostic findings, and the
temporal retention probe on MemoryS11) lives in [`docs/main.tex`](docs/main.tex).

## Repository layout

```
CS5180_Project/
├── config.py                     # Shared hyperparameter dataclass
├── config_presets.py             # doorkey / doorkey5x5 / memory presets
├── environment.yml               # Conda environment spec
├── verify_env.py                 # Post-install smoke test
├── common/                       # Env wrappers, rollout buffer, logger, eval
├── world_model/                  # World-model agent (train.py, model.py)
├── recurrent_ppo/                # Recurrent PPO baseline (train.py, model.py)
├── analysis/                     # CSV averaging, plotting, latent probe
├── scripts/                      # Miscellaneous (random baseline)
├── results/                      # Evaluation CSVs and figures cited in the report
├── runs/                         # Per-run config.json + trained .pt checkpoints
│                                 # (TensorBoard event files are gitignored)
└── docs/main.tex                 # Project report
```

## Setup

Install the conda environment (GPU build; PyTorch 12.4 CUDA):

```bash
conda env create -f environment.yml
conda activate cs5180_project
```

Verify everything is importable and that MiniGrid envs construct:

```bash
python verify_env.py
```

You should see `ALL GOOD!` at the bottom. If CUDA is not available the
script prints a warning; training on CPU works but is much slower.

## Reproducing the main experiments

Each training command writes:
- `runs/<algo>_<env>_s<seed>_<unix_ts>/events.out.tfevents.*` (TensorBoard scalars)
- `runs/<algo>_<env>_s<seed>_<unix_ts>/config.json` (exact hyperparameter snapshot)
- `runs/<algo>_<env>_s<seed>.pt` (final checkpoint, overwritten per seed)

### World-model agent (`world_model/train.py`)

Presets are defined in [`config_presets.py`](config_presets.py):
`doorkey` (8x8), `doorkey5x5` (small-grid sanity), `memory` (MemoryS11).

```bash
# DoorKey-5x5 sanity run (should solve in ~0.3 M steps)
python -m world_model.train --preset doorkey5x5 --seed 0  --timesteps 3000000
python -m world_model.train --preset doorkey5x5 --seed 42 --timesteps 3000000

# MemoryS11 (10 M steps per seed, ~few hours on a single modern GPU)
python -m world_model.train --preset memory --seed 0  --timesteps 10000000
python -m world_model.train --preset memory --seed 42 --timesteps 10000000

# DoorKey-8x8 (exploratory; see docs/main.tex for the actual per-seed
# hyperparameters used in the runs reported in the paper)
python -m world_model.train --preset doorkey --seed 0  --timesteps 10000000
python -m world_model.train --preset doorkey --seed 42 --timesteps 10000000
```

Any preset field can be overridden from the CLI, e.g. `--ent-coef 0.05`,
`--imagine-horizon 0`, `--tbptt-len 32`. Run
`python -m world_model.train --help` for the full list.

### Recurrent PPO baseline (`recurrent_ppo/train.py`)

```bash
python -m recurrent_ppo.train --env MiniGrid-DoorKey-5x5-v0 --seed 42 --timesteps 3000000
python -m recurrent_ppo.train --env MiniGrid-DoorKey-8x8-v0 --seed 42 --timesteps 10000000
python -m recurrent_ppo.train --env MiniGrid-MemoryS11-v0  --seed 42 --timesteps 10000000
```

### Viewing training curves

```bash
tensorboard --logdir runs
```

Each run directory also contains a `config.json` with the exact
hyperparameters used for that run.

## Analysis and figures

The scripts under [`analysis/`](analysis/) regenerate the averaged CSVs
and comparison figures cited in the report from the per-seed CSVs in
`results/`. They have no CLI arguments — just run them with the repo root
as the working directory.

```bash
# DoorKey-5x5: seed-average model-free and world-model curves, write
# averaged CSVs, save doorkey5x5_seed_mean_mf_vs_wm.png
python analysis/doorkey5x5_average_and_plot.py

# DoorKey-8x8: same, seed-averaged, aligned on shared step prefix
python analysis/doorkey8x8_plot_mf_wm_seed_avg.py

# MemoryS11: same
python analysis/memorys11_plot_mf_wm_seed_avg.py
```

### Temporal retention probe (MemoryS11)

Fits a linear probe on the frozen belief state of a trained
checkpoint, binned by step-within-episode. No extra training runs
required — it uses the checkpoint already saved in `runs/`.

```bash
# Trained checkpoint
python -m analysis.temporal_retention_probe \
    --checkpoint runs/world_model_MiniGrid-MemoryS11-v0_s0.pt \
    --hidden-dim 256 \
    --num-episodes 400 \
    --seed 20000 \
    --out results/MemoryS11/probe_s0_trained.csv

# Random-init sanity baseline
python -m analysis.temporal_retention_probe \
    --random-init \
    --hidden-dim 256 \
    --num-episodes 400 \
    --seed 20000 \
    --out results/MemoryS11/probe_random_baseline.csv
```

The output CSV reports per-bin test accuracy, test-set size, and
majority-class baseline — exactly the columns reproduced in
`docs/main.tex`, Table 4.

## Report

Primary write-up: [`docs/main.tex`](docs/main.tex) (AAAI 2026 style).
It compiles with `pdflatex main && bibtex main && pdflatex main && pdflatex main`
from inside `docs/`. Figures in the report read PNGs directly from
`results/`, so regenerating the analysis scripts above regenerates
everything the report cites.

## Hardware notes

All reported runs used a single consumer-grade NVIDIA GPU. A full 10 M-step
MemoryS11 run takes roughly 2–3 hours on an RTX-class GPU; the
DoorKey-5x5 sanity runs converge in under an hour.
