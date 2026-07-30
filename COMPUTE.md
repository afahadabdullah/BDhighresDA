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
ERA5-conditioned ensemble for each day. The output figure,
`data/processed/test_prediction_panels.png`, shows ERA5 precipitation, CHIRPS,
the ensemble-mean prediction, signed error, and ensemble spread. The companion
JSON report contains case-level ERA5 and model metrics. These target-selected
cases are a visual stress test, not an aggregate test-period skill estimate.

## Submit assimilation

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
