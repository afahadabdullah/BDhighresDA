# BDhighresDA

**Generative downscaling + data assimilation for daily rainfall over Bangladesh.**

ERA5 (0.25°) is downscaled to **0.05° (~5 km) daily precipitation** by a
conditional **flow-matching** generative model conditioned on ERA5 and GPM
IMERG, then corrected at inference time by **sparse BMD rain-gauge
observations** using score guidance — no retraining needed when the observing
network changes.

The approach follows [Manshausen et al. (2025, *JAMES*)](https://doi.org/10.1029/2024MS004505)
(score-based data assimilation of weather stations at km scale), with the
diffusion prior replaced by a rectified-flow model
([Lipman et al. 2023](https://arxiv.org/abs/2210.02747);
[Wetherell 2026](https://arxiv.org/abs/2606.00281)).

📖 **Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) first** — it contains
the full scientific design, the math linking flow matching to SDA guidance,
the data plan, and the experiment/ablation list.

---

## Pipeline at a glance

![BDhighresDA pipeline](docs/figures/pipeline.svg)

| Step | Script |
|---|---|
| 0. ERA5 predictors | `scripts/00_download_era5.py` |
| 1. CHIRPS target | `scripts/01_download_chirps.py` |
| 2. IMERG stage-1 obs | `scripts/02_download_imerg.py` |
| 3. Static fields (orography, mask, position) | `scripts/03_build_static.py` |
| 4. Regrid + pack to Zarr | `scripts/04_regrid_and_pack.py` |
| 5. Station QC + pseudo-stations | `scripts/05_prepare_stations.py` |
| 6. Normalisation stats | `scripts/06_compute_stats.py` |
| 7. Train (2 stages) | `scripts/train.py` |
| 8. Assimilate → product | `scripts/assimilate.py` |
| 9. Verify | `scripts/evaluate.py` |

## Quick start

```bash
# 0. environment
conda env create -f environment.yml && conda activate bdhires
pip install -e .

# 1. does everything work? (synthetic data, CPU, ~2 min, no downloads)
python scripts/smoke_test.py

# 2. data (this is the slow part: ~1.5 TB of ERA5 requests, days of queueing)
python scripts/00_download_era5.py  --start 1981 --end 2025 --out data/raw/era5
python scripts/01_download_chirps.py --start 1981 --end 2025 --out data/raw/chirps
python scripts/02_download_imerg.py  --start 2000 --end 2025 --out data/raw/imerg
python scripts/03_build_static.py --dem data/raw/dem/gmted.tif \
       --chirps data/raw/chirps/chirps_wide_2010.nc --out data/static/static_wide.nc
python scripts/04_regrid_and_pack.py --start 1981 --end 2025 --out data/processed/bd_wide.zarr
python scripts/06_compute_stats.py --zarr data/processed/bd_wide.zarr \
       --train-years 1981 2018 --transform log1p --out data/processed/stats.json
python scripts/05_prepare_stations.py --csv data/stations/bmd_daily_raw.csv \
       --zarr data/processed/bd_wide.zarr --out data/stations

# 3. train (2 x V100)
sbatch slurm/train_2xV100.sbatch          # runs stage A then stage B

# 4. tune the DA hyperparameters on pseudo-observations, THEN on real gauges
python scripts/evaluate.py --config configs/da.yaml --ckpt runs/stageB/final.pt \
       --start 2019-01-01 --end 2020-12-31 --tune --out results/tuning.json

# 5. produce the product
sbatch slurm/assimilate.sbatch            # array job, one year per task

# 6. verify against withheld gauges (3-fold CV) + baselines
python scripts/evaluate.py --config configs/da.yaml --ckpt runs/stageB/final.pt \
       --start 2021-01-01 --end 2023-12-31 --cv-folds 3 --out results/cv.json
```

## What you need to supply

| | |
|---|---|
| **CDS API key** | `~/.cdsapirc` for ERA5 ([how-to](https://cds.climate.copernicus.eu/how-to-api)) |
| **Earthdata login** | `~/.netrc` entry for IMERG (GES DISC) |
| **BMD gauge CSV** | `data/stations/bmd_daily_raw.csv`, columns `station_id,name,lat,lon,date,precip_mm` |
| **DEM** | GMTED2010 or SRTM over 84–97°E, 16–29°N. Optional but strongly recommended — orography is the most informative static channel over Bangladesh |

## Domains

| Grid | Extent | Size | Use |
|---|---|---|---|
| `wide` | 84.0–96.8°E, 16.0–28.8°N | 256×256 @ 0.05° | training (random 128×128 crops) |
| `bd` | 87.6–94.0°E, 20.3–26.7°N | 128×128 @ 0.05° | evaluation and the released product |

The wide domain exists because ~16,000 daily fields is a small dataset for a
generative model; random cropping multiplies the effective sample count and
exposes the model to a wider range of rainfall regimes. See §5 of the
methodology.

## Compute notes

* **2 × V100 (32 GB)**: `configs/train_v100.yaml`, batch 8/GPU. V100 is
  `sm_70` and has **no bf16 tensor cores**, so `utils/dist.amp_dtype()`
  automatically selects fp16 + `GradScaler`. Expect ~1–2 days for stage B.
* **H100**: `configs/train_h100.yaml`, bigger model, bf16, batch 32.
  Expect a few hours.
* Guided sampling backpropagates through the network, so a 16-member guided
  day costs ~2–3× an unguided one — budget ~10–30 s/day on one GPU.

## Repository layout

```
src/bdhires/
  grids.py          domain definitions (lat ASCENDING everywhere)
  transforms.py     precipitation transforms (log1p / sqrt / cbrt)
  models/unet.py    ADM-style U-Net velocity network
  models/flow.py    rectified flow: interpolant, loss, score<->velocity identities, EMA
  da/observation.py differentiable bilinear station operator, R, k-fold splits
  da/guidance.py    Gaussian obs likelihood + guidance gradient (SDA Eq. 3)
  da/sampler.py     Heun ODE sampler with guidance + Langevin correctors
  data/             Zarr dataset with random-crop augmentation; BMD station I/O
  eval/metrics.py   RMSE/MAE/CRPS/spread-skill/rank hist, FSS, SAL, POD/FAR/CSI/ETS
configs/            data + model + training + DA hyperparameters
scripts/            numbered data pipeline, then train / assimilate / evaluate
slurm/              ready-to-submit batch scripts
```

## Status

Scaffold complete and smoke-tested; no real data has been ingested yet.
Open items are tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

MIT (code). Note that CHIRPS, ERA5 and IMERG each carry their own
attribution requirements, and BMD gauge data is not redistributable without
BMD's permission — `data/` is gitignored.
