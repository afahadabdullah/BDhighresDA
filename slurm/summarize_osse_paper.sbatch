#!/bin/bash
#SBATCH --job-name=bdhires-osse-summary
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

ROOT="${OSSE_PAPER_ROOT:-data/processed/osse_paper}"
PRIMARY="${OSSE_PAPER_PRIMARY:-simultaneous_realistic_40}"

# One driver assembles every arm into the paper evidence set: ablation and
# downscaling/source-value tables (LaTeX), figure drafts, tidy CSVs, combined
# curve NPZ, and RESULTS.md.  Ensemble scores are not recomputed here; source
# attribution is derived from each arm's existing scale_summary.json.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/24_osse_paper_suite.py \
    --root "$ROOT" \
    --primary "$PRIMARY" \
    --out-dir "$ROOT/paper"

echo
echo "Paper artifacts in $ROOT/paper:"
ls -1 "$ROOT/paper"
