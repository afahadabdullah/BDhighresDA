# V3-SG implementation runbook

V3-SG uses a new CPC-edge-aligned archive. Do not point it at
`bd_wide_cpc.zarr`: that store contains CPC bilinearly interpolated to the
legacy 0.05-degree grid and is intentionally rejected by the V3 preparation
path.

## 1. Prepare aligned raw inputs

Use separate output directories so the 260-by-260 files cannot be confused
with the legacy 256-by-256 files.

```bash
python scripts/01_download_chirps.py \
  --start 1981 --end 2024 --grid wide_cpc \
  --out data/raw/chirps_v3sg

python scripts/00_download_era5.py \
  --start 1981 --end 2024 --grid wide_cpc \
  --out data/raw/era5_v3sg

python scripts/02b_download_cpc.py \
  --start 1981 --end 2024 --out data/raw/cpc --require-complete

python scripts/03_download_dem.py \
  --grid wide_cpc \
  --out data/raw/dem/copernicus_glo90_wide_cpc.nc

python scripts/03_build_static.py \
  --grid wide_cpc \
  --dem data/raw/dem/copernicus_glo90_wide_cpc.nc \
  --chirps data/raw/chirps_v3sg/chirps_wide_cpc_2010.nc \
  --out data/static/static_wide_cpc.nc
```

## 2. Build the paired target archive

```bash
python scripts/56_build_chirps_subgrid_targets.py \
  --chirps-glob 'data/raw/chirps_v3sg/chirps_wide_cpc_*.nc' \
  --cpc-glob 'data/raw/cpc/precip.*.nc' \
  --era5-glob 'data/raw/era5_v3sg/era5_daily_*.nc' \
  --static data/static/static_wide_cpc.nc \
  --out data/processed/cpc_v3_subgrid/wide_cpc.zarr
```

This is a two-pass, chunked writer. The first pass freezes training-period
amount and conditioning statistics; the second writes raw rainfall, paired
flow states and normalized conditions. Target dequantisation is seeded per
date, so changing `--chunk-days` cannot change the learned target.

## 3. Train the three phases

```bash
python scripts/57_train_subgrid_oracle.py \
  --config configs/train_h100_cpc_v3_subgrid_coarse.yaml

python scripts/57_train_subgrid_oracle.py \
  --config configs/train_h100_cpc_v3_subgrid_allocation.yaml

python scripts/57_train_subgrid_oracle.py \
  --config configs/train_h100_cpc_v3_subgrid_joint.yaml
```

The joint configuration loads both `best.pt` branch checkpoints strictly. A
channel mismatch or partial transfer is an error. Joint checkpoints record the
clean-coarse-context training probability; the coupled-oracle sampler refuses a
checkpoint where that probability is zero.

## 4. Preflight the scientific invariants

Run this before an expensive training or DA submission:

```bash
PYTHONPATH=src pytest -q tests/test_v3_subgrid.py
```

The tests cover CPC alignment, the 160-cell production halo, quantized crops,
coastal mass conservation, exact dry atoms with occurrence gradients,
chunk-invariant targets, literal branch transfer, joint gauge gradients,
0.4/0.5-degree uniform-footprint behavior and physical authority closure.

## 5. Evaluate a generated archive

A generated Zarr must contain `time` and one `(time, member, lat, lon)` array
per requested method. Optional `<method>_coarse_mm` arrays activate the direct
conservation check; optional latent-state arrays activate the physical
amount/allocation authority decomposition.

```bash
python scripts/58_evaluate_subgrid_prior.py \
  --target-store data/processed/cpc_v3_subgrid/wide_cpc.zarr \
  --sample-store data/processed/cpc_v3_subgrid/confirmation.zarr \
  --methods background,imerg_only,gauges_only,simultaneous_huber3 \
  --out-dir data/processed/cpc_v3_subgrid/evaluation
```

Subgrid anomaly scores remove each member's own area-weighted 0.5-degree mean.
This prevents coarse amount skill from being mislabeled as subgrid placement
skill.
