#!/bin/bash
#SBATCH --job-name=bdhires-osse-gauges-figs
#SBATCH --partition=grace-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A.out

# Figures for the gauges-only OSSE. CPU only: everything here reads the NPZ and
# JSON the GPU array already wrote.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${OSSE_GAUGE_ROOT:-data/processed/osse_gauges}"
PRIMARY="${OSSE_GAUGE_PRIMARY:-gauges_realistic_bmd}"
OUT="${OSSE_GAUGE_FIGURES:-docs/paper_figures_osse_gauges}"

# The tuning arm holds sixty configurations, not one, so it has no
# downscaling.json and discover_arms will not pick it up. That is correct: it
# belongs in its own table, not as a row beside single-configuration arms.
echo "Arms present under $ROOT:"
ls -1 "$ROOT" 2>/dev/null | sed 's/^/  /'

srun --ntasks=1 "$PYTHON_BIN" -u scripts/24_osse_paper_suite.py \
    --root "$ROOT" --primary "$PRIMARY" --out-dir "$ROOT/paper" || {
    echo "WARNING: scripts/24 failed; continuing to the paper figures." >&2
}

srun --ntasks=1 "$PYTHON_BIN" -u scripts/47_osse_paper_figures.py \
    --root "$ROOT" --primary "$PRIMARY" --out-dir "$OUT"

echo
echo "============================================================"
echo " Gauges-only OSSE figures: $OUT"
echo " Data behind every panel:  $OUT/data/*.csv"
echo
echo " The 60-combination tuning matrix is separate:"
echo "   $ROOT/gauges_tune_bmd/osse_report.json"
echo " Look for whether sigma_obs = 0.40 is near-optimal against a known"
echo " truth. On real data it was among the worst, and the claim was that the"
echo " prior's wet bias -- not the representativeness measurement -- was to"
echo " blame. This is the test of that claim."
echo "============================================================"
