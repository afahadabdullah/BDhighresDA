# V3-SG implementation runbook

V3-SG uses a new CPC-edge-aligned archive. Do not point it at
`bd_wide_cpc.zarr`: that store contains CPC bilinearly interpolated to the
legacy 0.05-degree grid and is intentionally rejected by the V3 preparation
path.

## 1. Reuse the existing raw inputs

No CHIRPS or DEM download is required. The V3 `wide_cpc` domain is the
240-by-240 inward CPC-aligned subset of the existing 256-by-256 `wide` domain.
Script 56 selects those CHIRPS cells exactly; it does not interpolate them.
The inward subset retains the complete frozen 160-by-160 Bangladesh production
canvas.

```bash
# These existing inputs are used directly:
ls data/raw/chirps/chirps_wide_*.nc
ls data/raw/cpc/precip.*.nc
ls data/raw/era5/era5_daily_*.nc
ls data/raw/dem/copernicus_glo90_wide.nc

# The pipeline creates this file automatically if it is absent. This manual
# command is only useful when static fields should be built separately.
python scripts/03_build_static.py \
  --grid wide_cpc \
  --dem data/raw/dem/copernicus_glo90_wide.nc \
  --chirps data/raw/chirps/chirps_wide_2010.nc \
  --out data/static/static_wide_cpc.nc
```

## 2. Build the paired target archive

```bash
python scripts/56_build_chirps_subgrid_targets.py \
  --chirps-glob 'data/raw/chirps/chirps_wide_*.nc' \
  --cpc-glob 'data/raw/cpc/precip.*.nc' \
  --era5-glob 'data/raw/era5/era5_daily_*.nc' \
  --static data/static/static_wide_cpc.nc \
  --start 1981-01-01 --end 2024-12-31 \
  --out data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr
```

The end date deliberately excludes the existing 2025 CHIRPS file so the
archive remains matched to the originally frozen 1981--2024 CPC/ERA5 period.

This is a two-pass, chunked writer. The first pass freezes training-period
positive-amount, wet-cell centred-log-allocation and conditioning statistics;
the second writes raw rainfall, paired flow states and normalized conditions.
Dry decoder-inactive channels receive neutral standardized values and a 0.05
relative loss weight, enough to define their marginal without letting dry cells
dominate training. Allocation intensity is clipped at +/-6 in standardized
latent units before the frozen mean/std are undone. Target dequantisation is
seeded per date, so changing `--chunk-days` cannot change the learned target.
The positive channel uses a proper weighted-mean denominator and occurrence
uses a valid-cell mean, so their aggregate configured balance does not vary
with seasonal wet fraction. A coarse block is wet when at least one valid fine
cell exceeds 0.1 mm/day; applying that per-cell threshold to the block mean
would erase isolated convective cells.
Occurrence remains trained across every valid block/cell.

The corrected archive schema is `cpc_v3_subgrid_v4`. Earlier V3 target archives
and checkpoints are intentionally rejected and must be moved aside before
restarting; they encode a different occurrence/loss/decoder contract. Preparation
also records `hard_threshold_oracle_ceiling`, the target-projection loss against
raw CHIRPS caused by the 0.1 mm/day hard wet threshold and decoder before any
model is trained.

## 3. Train the three phases

On Prism, submit the complete dependency chain from the repository root:

```bash
bash slurm/submit_v3_subgrid_pipeline.sh
```

This first runs the V3 regression suite on `grace-cpuonly`, sends preparation
to `grace-cpuonly` after the tests pass, starts the coarse and allocation GH200
jobs in parallel after preparation succeeds, and starts joint training only
after both branches succeed. Set `V3_RUN_TESTS=0` only for a deliberate rerun
after the same commit has already passed. A completed schema-v4 target archive
is reused.
Interrupted training resumes from `last.pt` only when the saved and requested
configs match exactly. If `data/static/static_wide_cpc.nc` is absent, the
preparation job builds it from an existing yearly CHIRPS file and
`data/raw/dem/copernicus_glo90_wide.nc`.

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/bdhires-v3-*.out
```

The equivalent phase-by-phase commands inside a compatible environment are:

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

Validation selects checkpoints over four deterministic 120-by-120 tiles that
cover the complete 240-by-240 domain and over every validation batch. It no
longer selects on one central crop or the first 32 batches.

### Diagnostic use of the superseded pre-v4 checkpoint

If the earlier `runs/prior_h100_cpc_v3_subgrid/joint/best.pt` completed, it can
be inspected without mixing it into the corrected experiment:

```bash
bash slurm/submit_v3_subgrid_legacy_diagnostic.sh
```

This samples May 1--5, 2022 on the aligned 160-cell Bangladesh canvas using
matched initial noise for an unguided background and a preliminary gauges-only
analysis. It saves ordinary rainfall maps, below-0.5-degree anomaly maps, the
physical ensembles and latent states, and separate assimilated/withheld gauge
scores under `data/processed/v3_legacy_diagnostic/may2022_5day`. The archive is
marked `legacy_pre_v4` and cannot be consumed by the corrected v4 evaluator.
The physical-space gauge settings are diagnostic, not selected V3 DA settings.
The completed legacy run uses schema `cpc_v3_subgrid_v2`; this diagnostic
replays its original raw-log-weight clip exactly rather than reinterpreting it
with the corrected v4 standardized-latent decoder.

## 4. Preflight the scientific invariants

Run this before an expensive training or DA submission. On the Prism login
node, submit the ARM test wrapper rather than trying to execute the ARM Python
binary directly:

```bash
sbatch slurm/v3_subgrid_tests.sbatch
```

The equivalent command inside a compatible ARM environment is:

```bash
PYTHONPATH=src pytest -q tests/test_v3_subgrid.py
```

The tests cover CPC alignment, the 160-cell production halo, quantized crops,
coastal mass conservation, exact dry atoms with occurrence gradients,
the drizzle representation ceiling, chunk-invariant normalized targets,
literal branch transfer, full Phase-2
coarse-corruption support, joint gauge gradients, 0.4/0.5-degree
uniform-footprint behavior, pure amount/allocation attribution and physical
authority closure.

## 5. Evaluate a generated archive

A generated Zarr must be written with
`bdhires.zarr_output.write_hierarchical_sample_zarr`. It stores `time`, one
`(time, member, lat, lon)` hard-terminal field per method, both latent states,
and optional `<method>_coarse_mm` arrays. The writer reopens the temporary
archive, hard-decodes every serialized coarse/allocation state with the stored
masks, areas and encoding, and checks that result against every physical field
before atomically marking it complete. Script 58 rejects an unverified,
pre-v3-sample-schema or manually assembled store.

```bash
python scripts/58_evaluate_subgrid_prior.py \
  --target-store data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr \
  --sample-store data/processed/cpc_v3_subgrid/confirmation.zarr \
  --methods background,imerg_only,gauges_only,simultaneous_huber3 \
  --out-dir data/processed/cpc_v3_subgrid/evaluation
```

Subgrid anomaly scores remove each member's own area-weighted 0.5-degree mean.
This prevents coarse amount skill from being mislabeled as subgrid placement
skill.
