#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-gauge-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-gauge-summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_SWEEP_ROOT:-data/processed/v2_gauge_da_sweep/ing2022_core}"
CURRENT="${V2_SWEEP_CURRENT:-guided_s0_t125}"
DUMPS=()
REPORTS=()
for fold in 0 1 2 3 4; do
    dump="$ROOT/fold${fold}.npz"
    report="$ROOT/fold${fold}.json"
    [[ -s "$dump" ]] || { echo "ERROR: missing $dump"; exit 1; }
    [[ -s "$report" ]] || { echo "ERROR: missing $report"; exit 1; }
    DUMPS+=("$dump")
    REPORTS+=("$report")
done

"$PYTHON_BIN" -u scripts/49_summarize_v2_gauge_sweep.py \
    --dumps "${DUMPS[@]}" \
    --reports "${REPORTS[@]}" \
    --baseline background \
    --current "$CURRENT" \
    --block-days "${V2_SWEEP_BLOCK_DAYS:-3}" \
    --n-resamples "${V2_SWEEP_BOOTSTRAPS:-10000}" \
    --out-json "$ROOT/method_selection.json" \
    --out-markdown "$ROOT/method_selection.md" \
    --out-plot "$ROOT/method_selection.png"

echo "CPC-v2 gauge DA selection complete."
echo "  table:  $ROOT/method_selection.md"
echo "  data:   $ROOT/method_selection.json"
echo "  figure: $ROOT/method_selection.png"
