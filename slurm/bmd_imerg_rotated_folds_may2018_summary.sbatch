#!/bin/bash
#SBATCH --job-name=bdhires-may-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs
CARTOPY_DIR="${CARTOPY_DATA_DIR:-data/static/cartopy}"
mkdir -p "$CARTOPY_DIR"

"$PYTHON_BIN" - <<'PY'
import cartopy
import matplotlib

print("Cartopy:", cartopy.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

EVALUATIONS=()
DUMPS=()
REPORTS=()
for fold in 0 1 2 3 4; do
    prefix="data/processed/bmd_imerg_offset_m1_fold${fold}_20180501_31"
    [[ -f "${prefix}_evaluation.json" ]] || { echo "ERROR: missing ${prefix}_evaluation.json"; exit 1; }
    [[ -f "${prefix}.npz" ]] || { echo "ERROR: missing ${prefix}.npz"; exit 1; }
    [[ -f "${prefix}.json" ]] || { echo "ERROR: missing ${prefix}.json"; exit 1; }
    EVALUATIONS+=("${prefix}_evaluation.json")
    DUMPS+=("${prefix}.npz")
    REPORTS+=("${prefix}.json")
done

srun --ntasks=1 "$PYTHON_BIN" -u scripts/20_summarize_bmd_rotated_folds.py \
    --evaluations "${EVALUATIONS[@]}" \
    --out-json data/processed/bmd_imerg_offset_m1_rotated_summary_20180501_31.json \
    --out-plot data/processed/bmd_imerg_offset_m1_rotated_summary_20180501_31.png

srun --ntasks=1 "$PYTHON_BIN" -u scripts/21_bmd_fullmonth_diagnostics.py \
    --dumps "${DUMPS[@]}" \
    --reports "${REPORTS[@]}" \
    --summary data/processed/bmd_imerg_offset_m1_rotated_summary_20180501_31.json \
    --out-verification data/processed/bmd_imerg_offset_m1_fullmonth_verification.png \
    --out-spatial data/processed/bmd_imerg_offset_m1_fullmonth_spatial_impact.png \
    --out-report data/processed/bmd_imerg_offset_m1_fullmonth_diagnostics.json \
    --cartopy-data-dir "$CARTOPY_DIR"
