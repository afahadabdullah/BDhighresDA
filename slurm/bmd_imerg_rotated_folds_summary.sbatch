#!/bin/bash
#SBATCH --job-name=bdhires-fold-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs

EVALUATIONS=()
for fold in 0 1 2 3 4; do
    path="data/processed/bmd_imerg_offset_m1_fold${fold}_20180501_05_evaluation.json"
    [[ -f "$path" ]] || { echo "ERROR: missing $path"; exit 1; }
    EVALUATIONS+=("$path")
done

srun --ntasks=1 "$PYTHON_BIN" -u scripts/20_summarize_bmd_rotated_folds.py \
    --evaluations "${EVALUATIONS[@]}" \
    --out-json data/processed/bmd_imerg_offset_m1_rotated_summary.json \
    --out-plot data/processed/bmd_imerg_offset_m1_rotated_summary.png
