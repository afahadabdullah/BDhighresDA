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

The combined experiment uses the existing remote inputs without moving or
renaming them:

```text
data/bmd/bmd.csv
data/bmd/Stations.csv
data/imerg/3B-DAY.MS.MRG.3IMERG.20180501-S000000-E235959.V07B.nc4
...
data/imerg/3B-DAY.MS.MRG.3IMERG.20180531-S000000-E235959.V07B.nc4
```

For the required five-day controlled test, submit from the repository root:

```bash
slurm/submit_bmd_imerg_controlled_5day.sh
```

This wrapper fixes the period to May 1–5 and writes `controlled` outputs, so it
cannot overwrite the completed unstable experiment. The general full-period
entry point remains `slurm/submit_bmd_imerg_example.sh` but should not be used
until the five-day gate passes.

Before requesting a GPU analysis, the job performs two strict preflights. It
converts the two legacy BMD files to the canonical daily table, and it requires
exactly one IMERG V07B granule for every date from May 1 through May 31. Each
granule must contain `precipitation`, `randomError`, and
`precipitation_cnt`. The regional preparation keeps exact 0.1-degree
footprints nested over the model's 0.05-degree grid and, by default, rejects a
footprint when fewer than 40 of 48 half-hourly estimates contributed.

The GPU stage evaluates five matched arms: background, gauges only, IMERG only,
simultaneous gauges+IMERG, and IMERG followed by a localized serial deterministic
ensemble square-root gauge update. The same six stations remain withheld in all
arms. By default only every third IMERG footprint in each direction enters the
likelihood, and its variance is inflated by the approximate residual correlation
area. With a three-cell error length this is about 6.3x after thinning. This is
an effective-sample approximation; it replaces the demonstrably invalid
assumption that thousands of correlated pixels are independent.

IMERG enters through a physical 2x2 block-mean operator. V07B daily
`precipitation` and `randomError` are both in mm/day; do not multiply either by
24. BMD values and model output are also mm/day, but spatial and temporal
support differ: IMERG is a 0.1-degree footprint, BMD is a point gauge, and the
BMD reporting-day boundary must still be checked against IMERG's 00–24 UTC day.
The native error is transformed, combined with an uncertainty floor and
representativeness, correlation-inflated, and spatially perturbed.

Controlled outputs are:

```text
data/processed/imerg_bd_controlled_20180501_05.nc
data/processed/imerg_bd_controlled_20180501_05_qc.json
data/processed/bmd_imerg_controlled_20180501_05.npz
data/processed/bmd_imerg_controlled_20180501_05.json
data/processed/bmd_imerg_controlled_20180501_05_diagnostics.png
data/processed/bmd_imerg_controlled_20180501_05_spatial.png
data/processed/bmd_imerg_controlled_20180501_05_chirps_spatial.png
```

The metric matrix, withheld-station scatter/rank histograms, and ten-column
spatial suite compare all five arms. The JSON also records sequential gauge
innovation RMSE before and after each update. Read withheld-BMD CRPS/RMSE and
the IMERG-only physical range first; a visually sharper map is not evidence of
improvement.

After the baseline controlled run, submit the requested satellite-weight and
localization sensitivity:

```bash
slurm/submit_bmd_imerg_sensitivity_5day.sh
```

This is a three-element GPU array for extra IMERG R multipliers 2, 4 and 8,
limited to two concurrent GPUs. Each element evaluates 75 and 100 km sequential
gauge localizations from the same generated IMERG ensemble, so localization
does not duplicate GPU sampling. Files are labelled `r2_l75-100`,
`r4_l75-100`, and `r8_l75-100`.

Every report includes `chirps_spatial_evaluation`, where CHIRPS is explicitly
the 0.05-degree gridded target and every background/DA ensemble is scored using
gridpoint CRPS, ensemble-mean RMSE/MAE/bias/correlation, spread/skill and 90%
coverage. Each case also writes `*_chirps_spatial.png`: period-mean fields,
mean-error maps and temporal-RMSE maps for CHIRPS, background, gauges-only,
IMERG-only, simultaneous, and IMERG-to-gauges products. These remain
training-period consistency evaluations, not independent validation.

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
