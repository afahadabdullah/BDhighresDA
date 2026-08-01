#!/bin/bash
#SBATCH --job-name=bdhires-imerg-chirps-align
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

# Diagnose date lag versus geolocation shift using an already completed BMD
# experiment dump. No GPU or resampling is needed.
set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

DUMP="${ALIGN_DUMP:-data/processed/bmd_imerg_controlled_20180501_05.npz}"
OUT_JSON="${ALIGN_OUT_JSON:-data/processed/imerg_chirps_alignment_20180501_05.json}"
OUT_PLOT="${ALIGN_OUT_PLOT:-data/processed/imerg_chirps_alignment_20180501_05.png}"
MAX_LAG="${ALIGN_MAX_LAG:-2}"
MAX_SHIFT="${ALIGN_MAX_SHIFT:-4}"

[[ -f "$DUMP" ]] || { echo "ERROR: required dump not found: $DUMP"; exit 1; }

srun --ntasks=1 "$PYTHON_BIN" -u scripts/18_diagnose_imerg_chirps_alignment.py \
    --dump "$DUMP" \
    --out-json "$OUT_JSON" \
    --out-plot "$OUT_PLOT" \
    --max-lag "$MAX_LAG" \
    --max-shift "$MAX_SHIFT"

echo "IMERG/CHIRPS alignment diagnostic complete."
echo "Metrics: $OUT_JSON"
echo "Plot: $OUT_PLOT"
