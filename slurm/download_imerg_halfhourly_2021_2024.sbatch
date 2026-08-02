#!/usr/bin/env bash
#SBATCH --job-name=imerg-hh-21-24
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module purge 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="${IMERG_PYTHON:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
command -v wget >/dev/null || { echo "ERROR: wget is not available"; exit 1; }
[[ -f "$HOME/.netrc" ]] || {
    echo "ERROR: $HOME/.netrc is missing. Add the Earthdata urs.earthdata.nasa.gov entry."
    exit 1
}
for path in scripts/02_download_imerg_halfhourly.py scripts/08_prepare_imerg_observations.py; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path; submit from the repository root"; exit 1; }
done

START_YEAR="${IMERG_START_YEAR:-2021}"
END_YEAR="${IMERG_END_YEAR:-2024}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This job must be submitted as a Slurm array}"
YEAR_COUNT=$((END_YEAR - START_YEAR + 1))
TASK_COUNT=$((YEAR_COUNT * 12))
if (( YEAR_COUNT < 1 )); then
    echo "ERROR: IMERG_END_YEAR must not precede IMERG_START_YEAR"
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
    echo "ERROR: array task $TASK_ID is outside 0-$((TASK_COUNT - 1))"
    exit 1
fi

YEAR=$((START_YEAR + TASK_ID / 12))
MONTH=$((TASK_ID % 12 + 1))
MONTH_KEY=$(printf '%04d%02d' "$YEAR" "$MONTH")
BMD_START=$(printf '%04d-%02d-01' "$YEAR" "$MONTH")
BMD_END="$($PYTHON_BIN -c 'import calendar, sys; y=int(sys.argv[1]); m=int(sys.argv[2]); print(f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}")' "$YEAR" "$MONTH")"
DATE_START="${BMD_START//-/}"
DATE_END="${BMD_END//-/}"

OUT_ROOT="${IMERG_HALFHOURLY_OUT:-data/imerg_halfhourly}"
OUT="$OUT_ROOT/$YEAR"
JOBS="${IMERG_DOWNLOAD_JOBS:-2}"
PREPARE_MONTHLY="${IMERG_PREPARE_MONTHLY:-1}"
mkdir -p "$OUT" data/processed logs

echo "Task $TASK_ID/$((TASK_COUNT - 1)): BMD dates $BMD_START through $BMD_END"
echo "Downloading IMERG V07B regional half-hours to $OUT with $JOBS workers"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/02_download_imerg_halfhourly.py \
    --bmd-start "$BMD_START" \
    --bmd-end "$BMD_END" \
    --end-hour-utc 3 \
    --bbox 20.3,87.6,26.7,94.0 \
    --jobs "$JOBS" \
    --out "$OUT"

if [[ "$PREPARE_MONTHLY" == "1" ]]; then
    echo "Preparing and validating exact BMD windows for $MONTH_KEY"
    export PYTHONPATH="$PWD/src"
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
        --input "$OUT" \
        --source-frequency half-hourly \
        --start "$BMD_START" \
        --end "$BMD_END" \
        --min-count 48 \
        --accumulation-end-hour-utc 3 \
        --out "data/processed/imerg_bd_aligned_${DATE_START}_${DATE_END}.nc" \
        --report "data/processed/imerg_bd_aligned_${DATE_START}_${DATE_END}_qc.json"
else
    echo "IMERG_PREPARE_MONTHLY=$PREPARE_MONTHLY; skipping monthly preparation"
fi

echo "IMERG $MONTH_KEY download and validation complete."
