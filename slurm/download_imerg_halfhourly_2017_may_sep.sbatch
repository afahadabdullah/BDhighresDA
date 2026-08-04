#!/usr/bin/env bash
#SBATCH --job-name=imerg-hh-2017-smoke
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=48:00:00
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

START="2017-05-01"
END="2017-09-30"
OUT="${IMERG_HALFHOURLY_OUT:-data/imerg_halfhourly/2017}"
JOBS="${IMERG_DOWNLOAD_JOBS:-4}"
ROOT="${BMD_2017_SMOKE_ROOT:-data/processed/bmd_imerg_smoke_2017_may_sep}"
PREPARED="$ROOT/imerg_aligned_20170501_20170930.nc"
QC="$ROOT/imerg_aligned_20170501_20170930_qc.json"
mkdir -p "$OUT" "$ROOT" logs

echo "2017 May-September process test: downloading 153 exact BMD windows"
echo "IMERG interval span is 2017-04-30 03:00 through 2017-09-30 02:30 UTC"
echo "Expected half-hourly granules: $((153 * 48))"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/02_download_imerg_halfhourly.py \
    --bmd-start "$START" \
    --bmd-end "$END" \
    --end-hour-utc 3 \
    --bbox 20.3,87.6,26.7,94.0 \
    --jobs "$JOBS" \
    --out "$OUT"

echo "Preparing and validating all exact 03:00-to-03:00 UTC windows"
export PYTHONPATH="$PWD/src"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
    --input "$OUT" \
    --source-frequency half-hourly \
    --start "$START" \
    --end "$END" \
    --min-count 48 \
    --accumulation-end-hour-utc 3 \
    --out "$PREPARED" \
    --report "$QC"

echo "IMERG 2017 May-September download and validation complete."
echo "Prepared observations: $PREPARED"
echo "QC: $QC"
