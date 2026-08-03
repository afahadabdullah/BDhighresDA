#!/bin/bash
#SBATCH --job-name=bdhires-osse-final-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${OSSE_FINAL_ROOT:-data/processed/osse_final_jja_2021_2024}"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/24_osse_paper_suite.py \
    --root "$ROOT" \
    --primary simultaneous_exact_bmd \
    --out-dir "$ROOT/paper"

echo
echo "Final OSSE paper artifacts in $ROOT/paper:"
ls -1 "$ROOT/paper"
