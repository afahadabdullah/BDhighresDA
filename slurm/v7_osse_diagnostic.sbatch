#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-osse
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
START="${V7_OSSE_START:-2022-05-01}"
END="${V7_OSSE_END:-2022-05-03}"
MEMBERS="${V7_OSSE_MEMBERS:-8}"
STEPS="${V7_OSSE_STEPS:-50}"
ARMS="${V7_OSSE_ARMS:-background,da_meso,da_fine,da_both}"
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
for required in "$MESO_ARCHIVE" "$MESO_STATS" "$SUBGRID" "$STATIONS" \
                scripts/72_v7_two_stage_osse.py; do
    [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

echo "V7 OSSE diagnostic"
echo "  dates:   $START through $END"
echo "  arms:    $ARMS"
echo "  sampling: $MEMBERS members, $STEPS Heun steps"
echo "  output:  $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
    --meso-checkpoint "$MESO_CKPT" \
    --allocation-checkpoint "$ALLOC_CKPT" \
    --meso-archive "$MESO_ARCHIVE" \
    --meso-stats "$MESO_STATS" \
    --subgrid-archive "$SUBGRID" \
    --stations "$STATIONS" \
    --start "$START" --end "$END" \
    --members "$MEMBERS" --n-steps "$STEPS" \
    --arms "$ARMS" \
    --out "$OUT" \
    "$@"

echo "V7 OSSE diagnostic complete: $OUT/v7_two_stage_osse.json"
