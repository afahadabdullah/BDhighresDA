# BDhighresDA

**Generative downscaling and score-guided data assimilation for daily rainfall over Bangladesh.**

Conditional **rectified-flow** generative prior downscales ERA5 (0.25°) and CPC (0.5°) to **0.05° (~5 km) daily precipitation**. At inference time, **GPM IMERG half-hourly satellite footprints** and **sparse BMD rain gauge observations** are assimilated via score guidance — enabling zero-shot DA without model retraining.

![BDhighresDA pipeline](docs/figures/pipeline.png)

---

## Accomplishments & Completed Features

- **Generative Flow-Matching Prior**: Trained rectified-flow U-Net models (`configs/train_h100_cpc.yaml`) on NVIDIA GH200 GPUs using ERA5 predictors and CPC conditioning over the `wide` (256×256 @ 0.05°) domain.
- **55,000+ IMERG Granules Ingested**: Built automated pipeline (`scripts/02_download_imerg_halfhourly.py`, `scripts/08_prepare_imerg_observations.py`) downloading and processing 2021–2024 half-hourly GPM IMERG V07B data into exact BMD reporting windows (03:00–03:00 UTC).
- **Expanded BMD Station Catalogue**: Parsed and integrated 42 BMD weather stations across Bangladesh (including 7 newly inherited stations: *Dimla, Rajarhat, Gopalgonj, Natrakona, Nikli, Tarash, Tetulia*) with high-coverage quality control (`scripts/05_convert_bmd_dir.py`).
- **5-Fold Rotated Spatial Cross-Validation**: Implemented disjoint spatial holdout evaluation (`slurm/bmd_imerg_rotated_folds_eval.sbatch`) testing 16 ensemble members across 2021–2024 monsoon seasons (37–38 active stations per year, >4,000 withheld station-days).
- **Automated Multi-Year Pooled Diagnostics**: Developed multi-year pooling script (`scripts/22_summarize_multiyear_bmd_eval.py`) that aggregates CRPS, RMSE, MAE, Bias, Fisher-pooled Correlation, and heavy-rain Brier scores into JSON, Markdown tables, and auto-generated 6-panel summary figures (`bmd_imerg_2021_2024_pooled_summary.png`).

---

## Pipeline Overview

| Step | Task | Script / Command |
|---|---|---|
| 0 | ERA5 Predictors | `scripts/00_download_era5.py` |
| 1 | CPC / CHIRPS Targets | `scripts/01_download_chirps.py`, `scripts/02b_download_cpc.py` |
| 2 | IMERG Half-Hourly Ingestion | `scripts/02_download_imerg_halfhourly.py` |
| 3 | Static Fields & DEM | `scripts/03_download_dem.py`, `scripts/03_build_static.py` |
| 4 | Zarr Packing & QC | `scripts/04_regrid_and_pack.py`, `scripts/04_check_alignment.py` |
| 5 | Station Conversion | `scripts/05_convert_bmd_dir.py` |
| 6 | Prior Training | `scripts/train.py` (`slurm/submit_train_cpc_gh200.sh`) |
| 7 | Multi-Year Evaluation | `slurm/submit_bmd_imerg_2021_2024_all.sh` |
| 8 | Multi-Year Summary & Plots | `scripts/22_summarize_multiyear_bmd_eval.py` |

---

## Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml && conda activate bdhires
pip install -e .
```

### 2. Submit Multi-Year Real BMD + IMERG Evaluation (2021–2024)
On HPC cluster (Grace Hopper nodes):
```bash
bash slurm/submit_bmd_imerg_2021_2024_all.sh
```
This automatically runs 5-fold cross-validation for 2021, 2022, 2023, 2024, and the full multi-year period, followed by auto-chained pooled summary generation.

### 3. Generate Multi-Year Summary & Figures
```bash
python scripts/22_summarize_multiyear_bmd_eval.py \
    --summaries data/processed/bmd_imerg_eval_2021_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2022_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2023_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2024_may_jun/rotated_summary.json \
    --out-json data/processed/bmd_imerg_2021_2024_pooled_summary.json \
    --out-markdown data/processed/bmd_imerg_2021_2024_pooled_summary.md \
    --out-plot data/processed/bmd_imerg_2021_2024_pooled_summary.png
```

---

## Domains

| Grid | Bounds | Size | Purpose |
|---|---|---|---|
| `wide` | 84.0–96.8°E, 16.0–28.8°N | 256×256 @ 0.05° | Offline training (random 128×128 crops) |
| `bd` | 87.6–94.0°E, 20.3–26.7°N | 128×128 @ 0.05° | Evaluation & production products over Bangladesh |

---

## Repository Layout

```
src/bdhires/
  grids.py          domain definitions (cell-centre, lat ascending)
  transforms.py     precipitation transforms (log1p / sqrt)
  models/unet.py    ADM-style U-Net velocity network
  models/flow.py    rectified flow interpolant, velocity/score identities, EMA
  bmd.py            BMD station directory parser & catalogue mapper
  imerg.py          half-hourly IMERG V07B ingestion & BMD window accumulation
  da/observation.py station + IMERG observation operators & covariance
  da/guidance.py    Gaussian likelihood guidance gradients
  da/sampler.py     ODE/SDE samplers with SDA guidance
  eval/metrics.py   RMSE, MAE, CRPS, Correlation, Brier Scores
scripts/            pipeline tools, training, evaluation, and multi-year summary
slurm/              HPC cluster sbatch & master submission bash wrappers
```

---

## License

MIT (code). CHIRPS, ERA5, and GPM IMERG carry their own data terms. BMD station data is not redistributable without permission.
