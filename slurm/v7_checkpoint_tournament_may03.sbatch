#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-ckpt
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Apple-to-apple May 3 checkpoint tournament for the selected V7 simultaneous
# arm.  It freezes the live best.pt files once, then evaluates all four old/new
# stage combinations with identical data, stations, seed, members and steps.

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

REFERENCE_ROOT="${V7_CKPT_REFERENCE_ROOT:-data/processed/v7_osse/20260820_1356}"
REFERENCE_JSON="$REFERENCE_ROOT/v7_two_stage_real.json"
FROZEN_MESO="$REFERENCE_ROOT/checkpoints/meso_frozen.pt"
FROZEN_ALLOC="$REFERENCE_ROOT/checkpoints/allocation_frozen.pt"
LATEST_MESO_SOURCE="${V7_CKPT_LATEST_MESO:-runs/v7/meso/best.pt}"
LATEST_ALLOC_SOURCE="${V7_CKPT_LATEST_ALLOCATION:-runs/v7/allocation/best.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
OUT="${V7_CKPT_OUT:-data/processed/v7_checkpoint_tournament_may03/$(date +%Y%m%d_%H%M)}"
SEED="${V7_CKPT_SEED:-20220503}"

for required in "$REFERENCE_JSON" "$FROZEN_MESO" "$FROZEN_ALLOC" \
                "$LATEST_MESO_SOURCE" "$LATEST_ALLOC_SOURCE" \
                "$REFERENCE_ROOT/bmd_daily.csv" "$MESO_ARCHIVE" "$MESO_STATS" \
                "$SUBGRID" scripts/72_v7_two_stage_osse.py \
                scripts/78_rank_v7_checkpoint_pairs.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

mkdir -p "$OUT/source_checkpoints" "$OUT/results"
# Freeze live best.pt once. Both trainers replace checkpoints atomically, so
# each copy is complete and all four runs use exactly the same captured files.
cp -p "$LATEST_MESO_SOURCE" "$OUT/source_checkpoints/latest_meso.pt"
cp -p "$LATEST_ALLOC_SOURCE" "$OUT/source_checkpoints/latest_allocation.pt"
LATEST_MESO="$OUT/source_checkpoints/latest_meso.pt"
LATEST_ALLOC="$OUT/source_checkpoints/latest_allocation.pt"

# Reuse the exact May 3 reference inputs and sampler horizon.  Support both the
# legacy string satellite field and the newer named-stream dictionary.
mapfile -t META < <("$PYTHON_BIN" - "$REFERENCE_JSON" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
dates = r.get("model_dates") or [
    day["date"] for arm in r["arms"].values() for day in arm.get("days", [])
]
if not dates:
    raise SystemExit("reference report has no model date")
satellite = r["satellite"]
if isinstance(satellite, dict):
    satellite = satellite.get("native")
if not isinstance(satellite, str):
    raise SystemExit(f"reference has no native IMERG path: {satellite!r}")
print(dates[0])
print(r["members"])
print(r["n_steps"])
print(satellite)
PY
)
START="${META[0]}"
MEMBERS="${META[1]}"
STEPS="${META[2]}"
IMERG="${META[3]}"
DAILY="$REFERENCE_ROOT/bmd_daily.csv"
[[ -f "$IMERG" ]] || { echo "ERROR: reference IMERG is missing: $IMERG" >&2; exit 1; }

cat > "$OUT/tournament_manifest.txt" <<EOF
date=$START
members=$MEMBERS
steps=$STEPS
seed=$SEED
arm=da_sim_r81
frozen_meso=$FROZEN_MESO
frozen_allocation=$FROZEN_ALLOC
latest_meso_source=$LATEST_MESO_SOURCE
latest_allocation_source=$LATEST_ALLOC_SOURCE
latest_meso_frozen=$LATEST_MESO
latest_allocation_frozen=$LATEST_ALLOC
imerg=$IMERG
stations=$DAILY
EOF

run_pair() {
    local label="$1" meso="$2" allocation="$3"
    local destination="$OUT/results/$label"
    mkdir -p "$destination"
    echo
    echo "======================================================================"
    echo "checkpoint pair: $label"
    echo "  meso: $meso"
    echo "  allocation: $allocation"
    echo "======================================================================"
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
        --meso-checkpoint "$meso" --allocation-checkpoint "$allocation" \
        --meso-archive "$MESO_ARCHIVE" --meso-stats "$MESO_STATS" \
        --subgrid-archive "$SUBGRID" --stations "$DAILY" --imerg "$IMERG" \
        --start "$START" --end "$START" \
        --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED" \
        --observations real --min-coverage 0.8 \
        --arms background --imerg-r-only 81 --no-plots \
        --station-dump "$destination/station_ensembles.npz" \
        --out "$destination"
}

run_pair frozen_frozen "$FROZEN_MESO" "$FROZEN_ALLOC"
run_pair latest_frozen "$LATEST_MESO" "$FROZEN_ALLOC"
run_pair frozen_latest "$FROZEN_MESO" "$LATEST_ALLOC"
run_pair latest_latest "$LATEST_MESO" "$LATEST_ALLOC"

"$PYTHON_BIN" -u scripts/78_rank_v7_checkpoint_pairs.py \
    --root "$OUT" \
    --out-json "$OUT/checkpoint_tournament.json" \
    --out-markdown "$OUT/checkpoint_tournament.md" \
    --seed "$SEED"

echo "Checkpoint tournament complete: $OUT/checkpoint_tournament.md"
