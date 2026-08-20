#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-osse
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# The V7 chain end to end, as an OSSE, against whatever both stages have learned
# so far.  Safe to fire repeatedly WHILE training runs: it copies both best.pt
# to a frozen path before sampling and never writes into either run directory.
#
# Four arms -- background / DA at 0.1 only / DA at 0.05 only / DA at both -- so a
# degradation can be attributed to a stage instead of just observed.  That
# attribution is the whole point; V5 cost a dozen runs for want of it.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

MESO_CKPT="${V7_OSSE_MESO_CKPT:-runs/v7/meso/best.pt}"
ALLOC_CKPT="${V7_OSSE_ALLOC_CKPT:-runs/v7/allocation/best.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
# Two DISTINCT files.  Stations.csv is the station CATALOG (metadata); the
# observation loader reads the canonical DAILY long-form CSV, which
# scripts/05_convert_bmd_dir.py builds from the per-station directory plus
# that catalog.  The v4 diagnostic already does exactly this -- reuse its
# converter rather than inventing a second route to the same file.
BMD_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
START="${V7_OSSE_START:-2022-05-03}"
END="${V7_OSSE_END:-2022-05-03}"
MEMBERS="${V7_OSSE_MEMBERS:-16}"
STEPS="${V7_OSSE_STEPS:-50}"
ARMS="${V7_OSSE_ARMS:-background,da_meso,da_fine,da_both}"
# A BMD-aligned IMERG netCDF at observation_factor 2 (0.1 degree, 03 UTC
# window).  Setting it enables the da_imerg / da_sim / da_sim_fine arms;
# the script refuses any file that is not on that grid or that window.
IMERG="${V7_OSSE_IMERG:-}"
imerg_args=()
if [[ -n "$IMERG" ]]; then
    [[ -f "$IMERG" ]] || { echo "ERROR: missing IMERG file $IMERG" >&2; exit 1; }
    imerg_args=(--imerg "$IMERG")
    echo "  imerg:   $IMERG"
fi
# osse: pseudo-gauges read CHIRPS, so station reporting is irrelevant and
# min-coverage 0 keeps the full network geometry.  real: assimilate and
# verify the ACTUAL reports, so a station that filed nothing must be
# dropped rather than carried as NaN -- and the gauge error is a real
# gauge error, which the script resolves from this same setting.
OBSERVATIONS="${V7_OSSE_OBSERVATIONS:-osse}"
if [[ "$OBSERVATIONS" == "real" ]]; then
    MIN_COVERAGE="${V7_OSSE_MIN_COVERAGE:-0.8}"
else
    MIN_COVERAGE="${V7_OSSE_MIN_COVERAGE:-0.0}"
fi
OUT="${V7_OSSE_OUT:-data/processed/v7_osse/$(date +%Y%m%d_%H%M)}"

# Both checkpoints must exist.  Mid-training is expected and fine; ABSENT is
# not, and the message should say which stage has not got there yet rather than
# surfacing as a torch.load traceback forty lines down.
for pair in "stage A:$MESO_CKPT" "stage B:$ALLOC_CKPT"; do
    label="${pair%%:*}"; path="${pair#*:}"
    [[ -f "$path" ]] || {
        echo "ERROR: $label has no checkpoint at $path" >&2
        echo "It has not reached its first validation epoch yet; wait and resubmit." >&2
        exit 1
    }
done
for required in "$MESO_ARCHIVE" "$MESO_STATS" "$SUBGRID" "$CATALOG" "$BMD_DIR" \
                scripts/05_convert_bmd_dir.py scripts/72_v7_two_stage_osse.py; do
    [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

mkdir -p "$OUT"
DAILY="$OUT/bmd_daily.csv"
if [[ ! -s "$DAILY" ]]; then
    # Widened by two days on each side: BMD day D is the 24h ending 03 UTC on
    # D, so model day C is read from BMD day C+1 and the window must contain it.
    CONV_START="$(date -u -d "$START -2 days" +%Y-%m-%d 2>/dev/null || date -u -v-2d -j -f %Y-%m-%d "$START" +%Y-%m-%d)"
    CONV_END="$(date -u -d "$END +2 days" +%Y-%m-%d 2>/dev/null || date -u -v+2d -j -f %Y-%m-%d "$END" +%Y-%m-%d)"
    echo "converting BMD $CONV_START..$CONV_END (widened for the day offset)"
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DIR" \
        --stations "$CATALOG" \
        --start "$CONV_START" --end "$CONV_END" \
        --out "$DAILY" \
        --summary "$OUT/bmd_stations.csv" \
        --report "$OUT/bmd_qc.json"
else
    echo "Reusing canonical BMD daily file: $DAILY"
fi

echo "V7 OSSE diagnostic"
echo "  dates:   $START through $END"
echo "  arms:    $ARMS"
echo "  sampling: $MEMBERS members, $STEPS Heun steps"
echo "  observations: $OBSERVATIONS   (min station coverage $MIN_COVERAGE)"
echo "  output:  $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
    --meso-checkpoint "$MESO_CKPT" \
    --allocation-checkpoint "$ALLOC_CKPT" \
    --meso-archive "$MESO_ARCHIVE" \
    --meso-stats "$MESO_STATS" \
    --subgrid-archive "$SUBGRID" \
    --stations "$DAILY" \
    --start "$START" --end "$END" \
    --members "$MEMBERS" --n-steps "$STEPS" \
    --arms "$ARMS" \
    "${imerg_args[@]}" \
    --observations "$OBSERVATIONS" \
    --min-coverage "$MIN_COVERAGE" \
    --out "$OUT" \
    "$@"

# The script names its output for the experiment it ran, not always _osse.
echo "V7 diagnostic complete: $OUT/v7_two_stage_${OBSERVATIONS}.json"
