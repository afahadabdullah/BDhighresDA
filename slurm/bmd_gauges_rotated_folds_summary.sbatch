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

LABEL="${BMD_EVAL_LABEL:-ing2022_V2}"
ROOT="data/processed/bmd_imerg_eval_${LABEL}"
V1_ROOT="${BMD_V1_REFERENCE_ROOT:-data/processed/bmd_imerg_eval_ing2022_RAW}"
WITHIN_OUT="$ROOT/gauges_vs_background"
COMPARE_OUT="$ROOT/v1_vs_v2"
BACKGROUND_OUT="$ROOT/v1_vs_v2_background"
mkdir -p "$MPLCONFIGDIR" "$WITHIN_OUT" "$COMPARE_OUT" "$BACKGROUND_OUT" logs

V2_DUMPS=()
ALL_DUMPS=()
for fold in 0 1 2 3 4; do
    v2_dump="$ROOT/fold${fold}.npz"
    v2_report="$ROOT/fold${fold}.json"
    v1_dump="$V1_ROOT/fold${fold}.npz"
    [[ -s "$v2_dump" ]] || { echo "ERROR: missing $v2_dump"; exit 1; }
    [[ -s "$v2_report" ]] || { echo "ERROR: missing $v2_report"; exit 1; }
    [[ -s "$v1_dump" ]] || {
        echo "ERROR: matched v1 reference is missing: $v1_dump" >&2
        echo "The direct comparison requires the ing2022_RAW May 1--10, 30-member folds." >&2
        exit 1
    }
    V2_DUMPS+=("$v2_dump")
    ALL_DUMPS+=("$v1_dump" "$v2_dump")
done

"$PYTHON_BIN" -u scripts/42_select_best_config.py \
    --dumps "${V2_DUMPS[@]}" \
    --arm gauges \
    --vs-arm background \
    --out-dir "$WITHIN_OUT"

# Primary checkpoint comparison. The cross-configuration contrast compares the
# v2 and v1 gauge analyses; --vs-arm additionally reports the assimilation
# increment relative to each checkpoint's own background.
"$PYTHON_BIN" -u scripts/42_select_best_config.py \
    --dumps "${ALL_DUMPS[@]}" \
    --arm gauges \
    --vs-arm background \
    --reference ing2022_RAW \
    --out-dir "$COMPARE_OUT"

# Separate the prior effect from the DA effect by comparing the two unguided
# backgrounds on the same dates, folds, member indices and withheld gauges.
"$PYTHON_BIN" -u scripts/42_select_best_config.py \
    --dumps "${ALL_DUMPS[@]}" \
    --arm background \
    --reference ing2022_RAW \
    --out-dir "$BACKGROUND_OUT"

echo "CPC-v2 real-BMD gauges-only summary complete ($LABEL)."
echo "Primary v1-v2 metrics: $COMPARE_OUT/config_selection.json"
echo "Primary v1-v2 figure: $COMPARE_OUT/config_selection.png"
echo "Background-only comparison: $BACKGROUND_OUT/config_selection.json"
echo "V2 gauges-vs-background: $WITHIN_OUT/config_selection.json"
