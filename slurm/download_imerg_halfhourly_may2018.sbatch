#!/usr/bin/env bash
#SBATCH --job-name=imerg-hh-may2018
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="${IMERG_PYTHON:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
command -v wget >/dev/null || { echo "ERROR: wget is not available"; exit 1; }
[[ -f "$HOME/.netrc" ]] || {
    echo "ERROR: $HOME/.netrc is missing. Add the Earthdata urs.earthdata.nasa.gov entry."
    exit 1
}
[[ -f scripts/02_download_imerg_halfhourly.py ]] || {
    echo "ERROR: submit from the BDhighresDA repository root"
    exit 1
}

OUT="${IMERG_HALFHOURLY_OUT:-data/imerg_halfhourly}"
JOBS="${IMERG_DOWNLOAD_JOBS:-2}"
mkdir -p "$OUT" data/processed logs

echo "Downloading exact May 2018 BMD windows to $OUT with $JOBS workers"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/02_download_imerg_halfhourly.py \
    --bmd-start 2018-05-01 \
    --bmd-end 2018-05-31 \
    --end-hour-utc 3 \
    --bbox 20.3,87.6,26.7,94.0 \
    --jobs "$JOBS" \
    --out "$OUT"

echo "Preparing and validating all 31 exact BMD accumulation windows"
export PYTHONPATH="$PWD/src"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
    --input "$OUT" \
    --source-frequency half-hourly \
    --start 2018-05-01 \
    --end 2018-05-31 \
    --min-count 48 \
    --accumulation-end-hour-utc 3 \
    --out data/processed/imerg_bd_aligned_20180501_31.nc \
    --report data/processed/imerg_bd_aligned_20180501_31_qc.json

echo "IMERG May 2018 download and exact-window validation complete."
