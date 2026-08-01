#!/bin/bash
#SBATCH --job-name=bdhires-bmd-imerg-plot
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

# Resume only the plotting stage after a completed BMD+IMERG analysis. This is
# intentionally CPU-only: never spend another GPU allocation to repair plots.

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

DUMP="${BMD_DUMP:-data/processed/bmd_imerg_may2018_example.npz}"
REPORT="${BMD_REPORT:-data/processed/bmd_imerg_may2018_example.json}"
DIAGNOSTICS="${BMD_DIAGNOSTICS:-data/processed/bmd_imerg_may2018_diagnostics.png}"
SPATIAL="${BMD_SPATIAL:-data/processed/bmd_imerg_may2018_spatial.png}"
CHIRPS_SPATIAL="${BMD_CHIRPS_SPATIAL:-data/processed/bmd_imerg_may2018_chirps_spatial.png}"

for required in "$DUMP" "$REPORT"; do
    [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done

srun --ntasks=1 "$PYTHON_BIN" -u scripts/16_plot_bmd_month_example.py \
    --dump "$DUMP" \
    --report "$REPORT" \
    --out-diagnostics "$DIAGNOSTICS" \
    --out-spatial "$SPATIAL" \
    --out-chirps-spatial "$CHIRPS_SPATIAL"

echo "Plot recovery complete."
echo "Diagnostics: $DIAGNOSTICS"
echo "Spatial cases: $SPATIAL"
echo "CHIRPS spatial evaluation: $CHIRPS_SPATIAL"
