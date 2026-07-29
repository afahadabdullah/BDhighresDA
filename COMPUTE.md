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

## Submit training

Use the wrapper from any directory. It resolves the repository root and
creates `logs` before calling `sbatch`, which is necessary because Slurm opens
the output file before the job body runs.

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

The completed checkpoint is `runs/prior_h100/final.pt`.

## Submit assimilation

After training has produced `runs/prior_h100/final.pt`, submit:

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
