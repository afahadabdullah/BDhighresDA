#!/usr/bin/env bash
#SBATCH --job-name=bdhires-gauge-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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

LABEL="${BMD_EVAL_LABEL:-v2_gauges_20220501_05}"
ROOT="data/processed/bmd_imerg_eval_${LABEL}"
OUT="$ROOT/gauges_vs_background"
mkdir -p "$MPLCONFIGDIR" "$OUT" logs

DUMPS=()
for fold in 0 1 2 3 4; do
    dump="$ROOT/fold${fold}.npz"
    report="$ROOT/fold${fold}.json"
    [[ -s "$dump" ]] || { echo "ERROR: missing $dump"; exit 1; }
    [[ -s "$report" ]] || { echo "ERROR: missing $report"; exit 1; }
    DUMPS+=("$dump")
done

"$PYTHON_BIN" -u scripts/42_select_best_config.py \
    --dumps "${DUMPS[@]}" \
    --arm gauges \
    --vs-arm background \
    --out-dir "$OUT"

echo "CPC-v2 real-BMD gauges-only summary complete ($LABEL)."
echo "Primary metrics: $OUT/config_selection.json"
echo "Comparison figure: $OUT/config_selection.png"
