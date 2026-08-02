# Compute guide

This repository's primary GPU workflow targets the NASA NCCS PRISM Grace
partition: one NVIDIA Grace Hopper/GH200 GPU on an `aarch64` node. The
scientific configuration remains `configs/train_h100.yaml`, its checkpoints
remain under `runs/prior_h100`, and data assimilation uses `configs/da.yaml`.

## PRISM ARM environment

The Slurm jobs use the existing ARM-native installation:

```text
Miniforge:   /home/afahad/nb/project/BDDA/miniforge3-aarch64
Environment: /home/afahad/nb/project/BDDA/envs/bdda-gh200
```

Initialize it in an interactive PRISM shell with:

```bash
source /home/afahad/nb/project/BDDA/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/afahad/nb/project/BDDA/envs/bdda-gh200
export PYTHONNOUSERSITE=1

cd /path/to/BDhighresDA
python -m pip install -e . --no-deps
```

`--no-deps` is intentional. The environment already contains compatible
ARM-native scientific and PyTorch packages; resolving or replacing them from
the project metadata could introduce incompatible x86 or user-site packages.
Do not install or update packages from within a batch job.

The Slurm scripts additionally purge modules, unset inherited `PYTHONHOME` and
`PYTHONPATH`, disable Python's user-site directory, activate the environment
by absolute prefix, and then set `PYTHONPATH` to the repository's `src`
directory. Do not load Miniforge, Anaconda, or CUDA modules, and do not source
`mamba.sh` in these jobs.

If the environment does not yet contain a GPU-enabled PyTorch build, install
it once from a Grace Hopper node:

```bash
slurm/setup_pytorch_gh200.sh
```

This modifies the existing `bdda-gh200` environment; it does not create a new
Conda environment. The setup follows the NCCS Grace Hopper example and
defaults to PyTorch 2.9.1 from the ARM CUDA 12.9 wheel index. It verifies CUDA
and Hopper compute capability before reporting `PYTORCH SETUP PASSED`. See the
[NCCS Prism ARM/PyTorch guidance](https://www.nccs.nasa.gov/using-prism/).

## Download CHIRPS

The CHIRPS download is a CPU-only Slurm array submitted to Prism's
`grace-cpuonly` partition by default. Each task downloads one yearly global
NetCDF file, immediately crops it to the repository's `wide` training domain,
writes `data/raw/chirps/chirps_wide_YEAR.nc`, and removes the global temporary
file. Interrupted downloads resume from `.part` files.

Submit from any directory:

```bash
/path/to/BDhighresDA/slurm/submit_download_chirps.sh
```

The default array covers 1981–2025 with at most two simultaneous downloads.
It uses zero-based Slurm indices (`0-44%2`) and maps those indices back to
calendar years inside the job, which keeps it compatible with clusters whose
maximum permitted array index is below 1981. Override the years, concurrency,
or partition through environment variables:

```bash
CHIRPS_START=1981 CHIRPS_END=2025 CHIRPS_MAX_PARALLEL=2 \
slurm/submit_download_chirps.sh
```

To test one year first:

```bash
CHIRPS_START=1981 CHIRPS_END=1981 \
slurm/submit_download_chirps.sh
```

On Prism's ARM CPU nodes, the job automatically uses the project's
`bdda-gh200` ARM-native Python environment. To override the partition or
interpreter, set `CHIRPS_PARTITION` or
`CHIRPS_PYTHON=/path/to/environment/bin/python`. Set `CHIRPS_OUT` to override
the repository-relative output directory.

## Download ERA5

ERA5 is read anonymously from Earthmover's free Icechunk v2 / Zarr v3 store in
the AWS Open Data Registry. No CDS key, Earthmover login or AWS credentials are
required. Each array task reads one year from the time-series-optimized
`single/temporal` group, crops it to 83.0–97.8°E, 15.0–29.8°N, aggregates the
six predictors to correctly aligned daily fields, and atomically writes
`data/raw/era5/era5_daily_YEAR.nc`.

Icechunk v2 requires Python 3.12, while the GH200 training environment uses
Python 3.11. Prism's Miniforge installation is ARM-native and cannot execute
directly on an x86_64 login node. Create the small dedicated download
environment once with the wrapper; it runs Conda on `grace-cpuonly`:

```bash
cd /home/afahad/project/BDDA/BDhighresDA
slurm/setup_earthmover_env.sh
```

The batch script finds `../envs/bdda-earthmover/bin/python` automatically and
verifies Python, Icechunk, PCodec, Zarr, Xarray, Dask and NetCDF4 before
accessing S3.

Test one year first:

```bash
ERA5_START=1981 ERA5_END=1981 slurm/submit_download_era5.sh
```

After the test succeeds, submit 1981–2025 with at most two simultaneous yearly
tasks:

```bash
ERA5_START=1981 ERA5_END=2025 ERA5_MAX_PARALLEL=2 \
slurm/submit_download_era5.sh
```

The default partition is `grace-cpuonly`. Override the output directory,
partition, or interpreter with `ERA5_OUT`, `ERA5_PARTITION`, or
`ERA5_PYTHON`. Set `ERA5_WORKERS` to change the Dask worker count inside each
four-CPU task. Keep array concurrency modest so the public S3 service and
cluster egress are not overloaded.

## Download DEM and build static fields

Copernicus DEM GLO-90 is read anonymously from its public AWS bucket; no
Copernicus account or AWS credentials are required. A single CPU-only job
downloads the one-degree Cloud Optimized GeoTIFF tiles intersecting the WIDE
domain, averages them directly to the project's 0.05-degree grid, and writes:

```text
data/raw/dem/copernicus_glo90_wide.nc
data/static/static_wide.nc
```

The second file contains the seven model-ready static channels: scaled
elevation, physical terrain slope, CHIRPS land-validity mask, and four absolute
position encodings. IMERG and gauge inputs are not used.

Submit from the repository root after at least one CHIRPS year has completed:

```bash
slurm/submit_dem_static.sh
```

The default mask source is `data/raw/chirps/chirps_wide_2010.nc`. If a
different year is available, set it explicitly:

```bash
CHIRPS_REFERENCE=data/raw/chirps/chirps_wide_1981.nc \
slurm/submit_dem_static.sh
```

The job runs on `grace-cpuonly` with four concurrent tile downloads. Override
those defaults with `DEM_PARTITION` and `DEM_JOBS`. Source 90 m tiles are
deleted after the compact regional product passes validation; set
`DEM_KEEP_TILES=1` to retain them.

## Pack training data and check daily alignment

After all 45 ERA5 and CHIRPS annual files and `data/static/static_wide.nc`
exist, submit:

```bash
slurm/submit_pack_training_data.sh
```

This CPU-only job creates the model-training store
`data/processed/bd_wide.zarr` from ERA5, CHIRPS and the static fields. IMERG
and gauges are deliberately excluded because they enter only during
assimilation. Packing is resumable by completed year; rerun the same command
after a timeout or node failure.

The same job then computes cosine-latitude-weighted regional daily rainfall
series and correlates CHIRPS day `t` against ERA5 day `t + lag` for lags
−2 through +2. It fails unless the maximum correlation occurs at lag zero and
writes the detailed result to:

```text
data/processed/alignment_qc.json
```

Override the default range or paths with `PACK_START`, `PACK_END`,
`PACK_ERA5`, `PACK_CHIRPS`, `PACK_STATIC`, `PACK_OUT`, and `PACK_QC_OUT`.
The default partition is `grace-cpuonly`; override it with `PACK_PARTITION`.

After alignment passes, compute normalization statistics from the 1981–2018
training period only:

```bash
slurm/submit_compute_stats.sh
```

This writes `data/processed/stats.json` using a fixed random sample of 1,500
training days and the `log1p` precipitation transform. The job refuses to run
unless `alignment_qc.json` reports a lag-zero pass. Override its inputs with
the `STATS_*` environment variables documented in `slurm/compute_stats.sbatch`.

Create and numerically validate the normalization diagnostics next:

```bash
slurm/submit_normalization_diagnostics.sh
```

The job writes one large figure,
`data/processed/normalization_diagnostics.png`, containing paired raw and
normalized maps and distributions for CHIRPS and all six ERA5 predictors,
followed by all seven static fields. Its companion
`normalization_diagnostics.json` checks finite values and approximate
zero-mean/unit-standard-deviation behavior on a deterministic training sample.
Both files are mandatory inputs to the GH200 preflight.

## Submit training

First run a short real-data preflight on one GH200:

```bash
/path/to/BDhighresDA/slurm/submit_preflight_training_gh200.sh
```

It uses the production batch size and model, exercises the multi-worker Zarr
loader, runs two optimizer/EMA steps and one validation batch, reports peak GPU
memory, and writes `data/processed/training_preflight.json`. It saves no model
checkpoint. Do not start the full run unless its log ends with
`PREFLIGHT PASSED`. The preflight refuses to run unless the normalization
diagnostic report passed and the figure checksum is unchanged.

Then use the training wrapper from any directory. It resolves the repository
root and creates `logs` before calling `sbatch`, which is necessary because
Slurm opens the output file before the job body runs:

```bash
/path/to/BDhighresDA/slurm/submit_train_gh200.sh
```

Equivalently, from the repository root:

```bash
cd /path/to/BDhighresDA
mkdir -p logs
sbatch slurm/train_h100.sbatch
```

Training automatically resumes from `runs/prior_h100/last.pt` when that file
exists. Disable automatic resumption for a submission with:

```bash
RESUME_IF_AVAILABLE=0 slurm/submit_train_gh200.sh
```

The training job verifies that the preflight passed on the current Git commit
with unchanged training configuration and statistics. The guard can be
disabled explicitly with `REQUIRE_TRAINING_PREFLIGHT=0`, but doing so is not
recommended.

Every validation cycle atomically updates `runs/prior_h100/last.pt` with the
latest model, EMA, optimizer, epoch, step, and global best validation score.
Use that checkpoint only to resume training. When validation improves,
`runs/prior_h100/best.pt` is atomically updated with the same complete state;
use its EMA weights for testing and production. The best score is computed
with those EMA weights and a fixed validation random seed, making scores
comparable across validation cycles. A successfully completed run also writes
`runs/prior_h100/final.pt`, which is the final epoch rather than necessarily
the best validation epoch.

## Plot held-out predictions

After `runs/prior_h100/best.pt` exists, submit the held-out background
diagnostic:

```bash
/path/to/BDhighresDA/slurm/submit_test_predictions.sh
```

It selects held-out days nearest the 50th, 90th, and 99th percentiles of
Bangladesh-domain mean CHIRPS precipitation and generates a 16-member
ERA5-conditioned ensemble for each day. It writes these diagnostic products:

- `data/processed/test_prediction_maps.png` contains clearly titled ERA5 input,
  CHIRPS target, ensemble-mean prediction, ERA5 error, model error, and
  ensemble-spread maps. Rainfall fields share a scale within each case, and
  both errors use the same symmetric scale.
- `data/processed/test_prediction_metrics.png` compares RMSE, deterministic
  versus ensemble CRPS, bias, spatial correlation, interval coverage, and
  maximum intensity.
- `data/processed/test_prediction_cases/DATE_qNN.png` provides a large 2x3
  spatial figure for every selected case, with longitude/latitude labels on
  every panel and the same six input/target/prediction/error/spread fields.

The companion `data/processed/test_prediction_report.json` contains the
underlying case metrics. Maps use a Cartopy Plate Carrée projection with
labeled longitude/latitude gridlines and Natural Earth 10 m coastlines,
national borders, and first-order boundaries. Boundary files are cached in
`data/static/cartopy`. These target-selected cases are a visual stress test,
not an aggregate test-period skill estimate.

## Run the CPC-to-CHIRPS OSSE

Before using real IMERG and BMD observations, run the CHIRPS observing-system
simulation experiment (OSSE) with the best CPC checkpoint:

```bash
slurm/submit_osse.sh
```

The default is the requested upper-bound experiment:

- `runs/prior_h100_cpc/best.pt` supplies the frozen generative prior. The OSSE
  reads the Zarr path, normalization statistics, residual definition, and
  conditioning-channel subset from that checkpoint, so it cannot accidentally
  mix the CPC model with ERA5 inputs.
- Held-out 2021–2025 July CHIRPS is the nature truth. Thirty days are evaluated.
- Exact physical 2×2 means of CHIRPS create the 0.1° pseudo-satellite product.
  The mean is taken in mm/day before applying the nonlinear precipitation
  transform.
- A spatially spread 40-station pseudo-network is drawn from CHIRPS. Thirty-two
  stations are assimilated and eight are withheld. With dense satellite data,
  those withheld points test unseen sub-footprint allocation; they are not an
  independent validation dataset.
- Sixteen members use spatially correlated satellite-observation perturbations
  and independent gauge perturbations to diagnose posterior spread.

The job writes the raw ensemble dump plus four complementary plotting suites:

```text
data/processed/osse_combined.png
data/processed/da_diagnostics.png
data/processed/da_impact_cases.png
data/processed/da_impact_aggregate.png
data/processed/osse_reconstruction_maps.png
data/processed/osse_metric_matrix.png
```

The diagnostics include reconstruction/error/spread spatial maps, rank
histograms, spread–skill, CRPS by intensity, reliability, fractions skill
score by scale, power spectra, increment-versus-station-distance, normalized
innovations, daily metric matrices, and a dedicated 0.05° subgrid matrix. Read
`data/processed/osse_scale_summary.json` before deciding whether this prior is
good enough for real DA. Improvement at 0.1° alone is not sufficient because
that scale is directly observed; require improved subgrid RMSE/correlation and
approximately nominal ensemble coverage as well.

This CHIRPS-on-CHIRPS OSSE is deliberately optimistic: pseudo-observations and
nature truth share one product and therefore omit real IMERG/BMD biases,
timing errors, and representativeness differences. Passing it is necessary,
not sufficient. The next gate is a real-observation experiment with IMERG bias
correction fitted without test-period leakage and independent BMD cross-validation.

Useful overrides include:

```bash
OSSE_DAYS=60 OSSE_MEMBERS=24 OSSE_WITHHOLD=0.25 slurm/submit_osse.sh
OSSE_CKPT=runs/prior_h100_cpc/epoch_0040.pt slurm/submit_osse.sh
```

## Run the May 2018 real-BMD process example

The historical BMD files on Prism are expected at:

```text
data/bmd/bmd.csv
data/bmd/Stations.csv
```

The rainfall file is a station-month matrix with day-of-month columns and
`***` missing values, not the canonical long-form table expected by the DA
loader. The dedicated job converts it, preserves the source files, performs
the real-gauge analysis, and creates the evaluation plots:

```bash
slurm/submit_bmd_example.sh
```

May 2018 was selected because it provides 930 valid observations from 30
stations, a 54.7% wet-station-day fraction, and events up to 179 mm/day. June
has only 23 stations and July only 16. Six geographically spread stations are
withheld and never enter the likelihood; all headline scores use those gauges.

Outputs are:

```text
data/processed/bmd_daily_may2018.csv
data/processed/bmd_stations_may2018.csv
data/processed/bmd_qc_may2018.json
data/processed/bmd_may2018_example.npz
data/processed/bmd_may2018_example.json
data/processed/bmd_may2018_diagnostics.png
data/processed/bmd_may2018_spatial.png
```

This is deliberately a process gate, not the paper's final validation: 2018 is
inside the CPC checkpoint's 1981-2018 training period. It tests the legacy BMD
conversion, physical observation operator, perturbed-observation guidance,
spatial withholding, ensemble calibration, and plot suite. The first run is
gauge-only because the packed prior dataset does not yet contain an ingestible,
bias-corrected real IMERG observation stream. After this passes, add real IMERG
and repeat with temporally held-out BMD data or a retrained prior whose test
period overlaps the BMD archive.

Do not choose a later checkpoint merely because it trained longer: use the
checkpoint with the lowest fixed-case validation CRPS (currently the `best.pt`
selection, observed near epoch 40 in the reported run).

## Run the May 2018 BMD + real-IMERG experiment

The corrected combined experiment uses the BMD archive day as a 24-hour
window ending at 03:00 UTC. Keep the Earthdata half-hourly subset files under
one directory; no renaming is required:

```text
data/bmd/bmd.csv
data/bmd/Stations.csv
data/imerg_halfhourly/3B-HHR.MS.MRG.3IMERG.20180430-S030000-E032959.0180.V07B.HDF5.SUB.nc4
...
data/imerg_halfhourly/3B-HHR.MS.MRG.3IMERG.20180531-S023000-E025959.0150.V07B.HDF5.SUB.nc4
```

Download all exact May 2018 windows on a CPU node with:

```bash
slurm/submit_download_imerg_halfhourly_may2018.sh
```

This requests 1,488 regional subset granules spanning interval starts
2018-04-30 03:00 through 2018-05-31 02:30 UTC. The downloader uses explicit
`wget -O` destinations, avoiding the GES DISC "name is too long" query-string
filename, and skips existing files that pass a NetCDF/HDF signature check.
Rerunning the same command therefore resumes an interrupted download. It also
prepares and strictly validates all 31 reporting windows, writing:

```text
data/processed/imerg_bd_aligned_20180501_31.nc
data/processed/imerg_bd_aligned_20180501_31_qc.json
```

### Download the 2021--2024 half-hourly IMERG record

Use the monthly Slurm array rather than one four-year job:

```bash
slurm/submit_download_imerg_halfhourly_2021_2024.sh
```

The default submission creates 48 monthly CPU tasks with at most two active at
once. Together they request 70,128 regional V07B granules for 1,461 BMD archive
days. The exact half-hour starts run from 2020-12-31 03:00 UTC through
2024-12-31 02:30 UTC because every archive day is the end of its 03:00-to-03:00
UTC accumulation. Files are divided among `data/imerg_halfhourly/2021` through
`data/imerg_halfhourly/2024`; the recursive IMERG loader treats these as one
inventory.

Every task is resumable: valid NetCDF/HDF responses are skipped and incomplete
`.part` files are replaced. After downloading a month, the task requires all 48
half-hours per BMD day and writes:

```text
data/processed/imerg_bd_aligned_YYYYMM01_YYYYMMDD.nc
data/processed/imerg_bd_aligned_YYYYMM01_YYYYMMDD_qc.json
```

The default two array tasks with two workers each cap the request rate at four
concurrent GES DISC subset calls. Lower that rate if Earthdata begins returning
transient failures:

```bash
IMERG_ARRAY_CONCURRENCY=1 IMERG_DOWNLOAD_JOBS=2 \
  slurm/submit_download_imerg_halfhourly_2021_2024.sh
```

The years are configurable, and monthly preparation can be disabled when only
the source granules are wanted:

```bash
IMERG_START_YEAR=2022 IMERG_END_YEAR=2023 IMERG_PREPARE_MONTHLY=0 \
  slurm/submit_download_imerg_halfhourly_2021_2024.sh
```

Rerun the same submission after an interrupted array; completed granules will
be scanned and skipped. Do not apply another one-day shift to these
observations. The preceding UTC date already appears wherever required by the
BMD 03:00 UTC reporting window.

This exact observation window already uses part of the preceding UTC calendar
day. Do not shift IMERG back by another day: for BMD date `D`, use the 48
half-hours from `D-1 03:00` through `D 02:30` UTC. A separate `D-1` sensitivity
is appropriate only for the checkpoint's calendar-day CPC and ERA5 prior
fields.

For the required five-day controlled test, submit from the repository root:

```bash
slurm/submit_bmd_imerg_controlled_5day.sh
```

This wrapper fixes the period to May 1–5 and writes `controlled` outputs, so it
cannot overwrite the completed unstable experiment. The general full-period
entry point remains `slurm/submit_bmd_imerg_example.sh` but should not be used
until the five-day gate passes.

Before requesting a GPU analysis, the job performs two strict preflights. It
converts the two legacy BMD files to the canonical daily table, then requires
48 consecutive IMERG V07B half-hourly granules for every BMD date. For BMD
date `D`, the first interval starts at `D-1 03:00 UTC` and the final interval
starts at `D 02:30 UTC`. Missing or duplicate intervals are fatal. The
half-hourly `precipitation` and `randomError` variables must be in mm/hr. The
regional preparation keeps exact 0.1-degree footprints nested over the
model's 0.05-degree grid and, by default, requires all 48 values at each
footprint.

The GPU stage evaluates four matched arms: background, gauges only, IMERG only,
and simultaneous gauges+IMERG. The sequential IMERG-to-gauges update was
retired after it failed the withheld-BMD probabilistic and deterministic gates.
The same six stations remain withheld in all active arms. By default only every third IMERG footprint in each direction enters the
likelihood, and its variance is inflated by the approximate residual correlation
area. With a three-cell error length this is about 6.3x after thinning. This is
an effective-sample approximation; it replaces the demonstrably invalid
assumption that thousands of correlated pixels are independent.

IMERG enters through a physical 2x2 block-mean operator. Half-hourly
`precipitation` is a rate in mm/hr, so each BMD-aligned daily depth is
`sum(rate * 0.5 hr)`. Half-hourly `randomError` is converted to depth and
accumulated in quadrature as an independence baseline; remaining temporal and
spatial dependence must be assessed with the existing DA error-inflation
sensitivity.
The prepared variables, BMD values and model output are all in mm/day. IMERG
is a 0.1-degree footprint while BMD is a point gauge, so spatial
representativeness error remains necessary.

Controlled outputs are:

```text
data/processed/imerg_bd_aligned_20180501_05.nc
data/processed/imerg_bd_aligned_20180501_05_qc.json
data/processed/bmd_imerg_aligned_20180501_05.npz
data/processed/bmd_imerg_aligned_20180501_05.json
data/processed/bmd_imerg_aligned_20180501_05_evaluation.json
data/processed/bmd_imerg_aligned_20180501_05_diagnostics.png
data/processed/bmd_imerg_aligned_20180501_05_events.png
data/processed/bmd_imerg_aligned_20180501_05_station_comparison.png
data/processed/bmd_imerg_aligned_20180501_05_spatial.png
data/processed/bmd_imerg_aligned_20180501_05_intercomparison.png
```

The metric matrix, withheld-station scatter/rank histograms, and nine-column
spatial suite compare the four active arms. Read withheld-BMD CRPS/RMSE and
the IMERG-only physical range first; a visually sharper map is not evidence of
improvement.

The plotting stage now treats the identical withheld BMD station-days as the
only primary reference. It collocates raw CPC, raw IMERG, CHIRPS, an IDW gauge
baseline, the background ensemble mean, and every DA ensemble mean at those
stations. Probabilistic DA rows additionally receive CRPS/CRPSS, rank,
50/80/90-percent coverage, Brier/reliability and threshold CSI diagnostics.
CHIRPS is retained only in a plot explicitly labelled as a non-independent
gridded-product intercomparison.

Completed controlled and sensitivity dumps can all be re-evaluated without a
GPU or a new DA run:

```bash
slurm/submit_bmd_imerg_replot_all_5day.sh
```

The CPU job skips unavailable cases, produces `*_evaluation.json`,
`*_diagnostics.png`, `*_events.png`, `*_station_comparison.png`,
`*_spatial.png`, and `*_intercomparison.png` for each available dump, then
writes the cross-case selection products:

```text
data/processed/bmd_imerg_5day_method_selection.json
data/processed/bmd_imerg_5day_method_selection.png
```

The fusion gate is strict: simultaneous DA must have lower withheld-BMD CRPS
than gauges-only. Five days and one station split remain a
process gate; they cannot support a final product-skill claim.

### CPC/ERA5 background timing sensitivity

The BMD observation labelled day `D` covers previous-day 03:00 through day
`D` 03:00 UTC. The checkpoint's daily CPC condition and five ERA5 state means
are calendar-day fields. Run this matched sensitivity before interpreting the
aligned five-day assimilation as a timing-correct result:

```bash
slurm/submit_bmd_imerg_offset_m1_5day.sh
```

The observation side is unchanged: BMD and half-hourly IMERG still cover
2018-05-01 through 2018-05-05 using exact 03:00-to-03:00 windows. Only the
complete checkpoint prior moves back one day, from 2018-05-01--05 to
2018-04-30--2018-05-04. CPC, CPC validity, all ERA5 channels, the CPC residual
base and seasonal encoding move together. CHIRPS is not an input; its
observation-date field stays fixed as contextual intercomparison only. The
offset-0 and offset-1 runs also use the same station holdout and
observation-date-based random seeds.

The separate outputs begin with:

```text
data/processed/bmd_imerg_aligned_offset_m1_20180501_05
```

Compare its withheld-BMD CRPS, RMSE, correlation, spread/skill, coverage and
day/station matrices against `bmd_imerg_aligned_20180501_05`. Select the timing
by the fused method's performance across rotated station folds; do not select
it from CHIRPS agreement or one five-day aggregate alone.

### Rotated spatial holdout gate

After accepting the previous-day CPC/ERA5 alignment, rotate the BMD holdout
before tuning the satellite weight:

```bash
slurm/submit_bmd_imerg_rotated_folds_5day.sh
```

This submits a five-element GPU array, limited to two concurrent GPUs. The 30
stations are partitioned into five deterministic, geographically spread,
disjoint folds; every station is withheld exactly once. All folds use the
same May 1--5 BMD/IMERG observation windows, `BACKGROUND_DAY_OFFSET=-1`, model
checkpoint, IMERG stride/R, sampler configuration and observation-date seeds.
Only the six withheld stations rotate.

A dependent CPU job starts automatically after all five folds succeed and
writes:

```text
data/processed/bmd_imerg_offset_m1_rotated_summary.json
data/processed/bmd_imerg_offset_m1_rotated_summary.png
```

The simultaneous method passes only when its pooled withheld-BMD CRPS is below
gauges-only and it wins at least half the folds. Otherwise gauges-only remains
the selected DA product. The retired sequential method is excluded even when
replotting older dumps.

Only after this fold gate should an optional IMERG-R sensitivity be considered:

```bash
slurm/submit_bmd_imerg_sensitivity_5day.sh
```

### Full-May rotated spatial holdout

Once the half-hourly download and its 31-window validation finish, extend the
same frozen four-arm experiment to every day in May:

```bash
slurm/submit_bmd_imerg_rotated_folds_may2018.sh
```

The five GPU array tasks each evaluate one disjoint BMD station fold over
2018-05-01--31. They retain `BACKGROUND_DAY_OFFSET=-1`, 16 members, IMERG
stride 3, the native-error 6.3x correlation inflation, and exact 03:00-to-03:00
UTC observation windows. The active arms remain background, gauges only,
IMERG only and simultaneous; the retired sequential method is absent. Fold
products end in `_20180501_31`, so the five-day results are not overwritten.
A dependent CPU job produces:

```text
data/processed/bmd_imerg_offset_m1_rotated_summary_20180501_31.json
data/processed/bmd_imerg_offset_m1_rotated_summary_20180501_31.png
```

It also generates two additional pooled diagnostics inspired by the OSSE
verification suite:

```text
data/processed/bmd_imerg_offset_m1_fullmonth_verification.png
data/processed/bmd_imerg_offset_m1_fullmonth_spatial_impact.png
data/processed/bmd_imerg_offset_m1_fullmonth_diagnostics.json
```

The verification figure includes pooled withheld-BMD rank histograms,
spread-skill, CRPS by observed intensity, reliability, event Brier scores,
spectral structure, increment reach, and normalised gauge/IMERG innovations.
The spatial figure maps increments and spread changes averaged across folds;
it does not call those maps error reduction because real observations provide
no independent gridded 0.05-degree truth. Its station map is the only spatial
skill panel and uses each BMD station when it is withheld from its fold. All
six spatial panels use a Cartopy Plate Carree map with 10 m Natural Earth
coastlines, country borders and first-order administrative boundaries. Cartopy
keeps those files in `data/static/cartopy` so subsequent summary jobs reuse the
same boundary cache.

This is the larger development gate. Because 2018 remains inside the prior
checkpoint's training years, final skill still requires retraining with 2018
excluded before treating May 2018 as independent validation.

That optional array now tests only extra IMERG R multipliers 2, 4 and 8 with
the offset-minus-one background; it no longer runs localization or sequential
updates.

Legacy run reports still contain `chirps_spatial_evaluation` for reproducibility,
but it is no longer used for method selection. Newly generated plots do not
call CHIRPS a target or label differences from it as model error.

If IMERG and CHIRPS features look displaced, do not shift either product by
eye. Run the footprint-matched timing/geolocation diagnostic on the completed
five-day dump:

```bash
slurm/submit_imerg_chirps_alignment.sh
```

It first averages each 2x2 group of 0.05-degree CHIRPS cells into the exact
0.1-degree IMERG support, then searches +/-2 days and +/-4 coarse cells. In the
report, positive `lag_days` compares CHIRPS day `t` with IMERG day `t+lag`;
positive `dx_cells` shifts IMERG east and positive `dy_cells` shifts it north.
The output files are
`data/processed/imerg_chirps_alignment_20180501_05.{json,png}`. Five days can
identify an obvious ingestion error, but a proposed timestamp or coordinate
change must be reproduced over at least a full month.

If assimilation completed and only the plotting stage failed, preserve the
expensive NPZ and resume on a CPU node:

```bash
slurm/submit_bmd_imerg_plot.sh
```

This is still a process experiment. The CPC prior was trained through 2018;
IMERG Final is gauge-adjusted; CHIRPS is gauge-based; and this first bounded
run does not fit an IMERG-to-reference bias correction. The next scientific
gate is rotated station withholding, a bias-correction fit outside the test
period, and evaluation in years excluded from prior training.

## Submit real-data assimilation

After training has produced `runs/prior_h100/best.pt`, submit:

```bash
/path/to/BDhighresDA/slurm/submit_assimilate_gh200.sh
```

Or, from the repository root:

```bash
mkdir -p logs
sbatch slurm/assimilate.sbatch
```

The array covers 2020–2025 and is limited to two simultaneous GH200 jobs by
default (`%2`). Change the array concurrency only when the allocation and
queue policy permit it. Outputs are written to
`data/processed/bdhires_analysis_YEAR.nc`.

## Preflight checks

Both GH200 jobs fail before running project code unless all of the following
are true:

- the node architecture is `aarch64`;
- Python comes from `/envs/bdda-gh200/`;
- SciPy is not imported from `~/.local`;
- PyTorch can access the allocated GPU.

The preflight also prints the Python executable, package locations, CUDA
availability, GPU model, and GPU memory to the job log.

## Monitor and cancel

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -f logs/bdhires-gh200-JOB_ID.out
```

Assimilation logs use the form
`logs/bdhires-da-gh200-ARRAY_JOB_ID_ARRAY_INDEX.out`.

Cancel a job or an entire array with:

```bash
scancel JOB_ID
```

## x86-64 V100 alternative

`slurm/train_2xV100.sbatch` is retained for PRISM x86-64 V100 nodes. It uses
two GPUs and a separate x86/CUDA Conda environment. Never activate or reuse
the `bdda-gh200` ARM environment with that script.
