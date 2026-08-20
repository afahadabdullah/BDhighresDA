#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-imerg-scale
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# V7-only matched IMERG ingestion sweep. Native 0.1-degree and S04 0.4-degree
# observations come from the same BMD-windowed source and are tested with fixed
# checkpoints, station split, seed, ensemble size, and sampling steps.

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

START="${V7_INGEST_START:-2022-05-01}"
END="${V7_INGEST_END:-2022-05-10}"
MEMBERS="${V7_INGEST_MEMBERS:-16}"
STEPS="${V7_INGEST_STEPS:-50}"
SEED="${V7_INGEST_SEED:-20220503}"
OUT="${V7_INGEST_OUT:-data/processed/v7_imerg_ingestion_may2022/$(date +%Y%m%d_%H%M)}"

V7_REFERENCE_ROOT="${V7_INGEST_REFERENCE_ROOT:-data/processed/v7_osse/20260820_1356}"
MESO_CKPT="${V7_INGEST_MESO_CKPT:-$V7_REFERENCE_ROOT/checkpoints/meso_frozen.pt}"
ALLOC_CKPT="${V7_INGEST_ALLOC_CKPT:-$V7_REFERENCE_ROOT/checkpoints/allocation_frozen.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
IMERG_SOURCE="${V7_INGEST_IMERG_SOURCE:-data/processed/imerg_bd_aligned_*.nc}"
if [[ "$IMERG_SOURCE" == /path/to/* ]]; then
    echo "WARNING: ignoring placeholder IMERG source: $IMERG_SOURCE" >&2
    IMERG_SOURCE="data/processed/imerg_bd_aligned_*.nc"
fi
BMD_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

for required in "$MESO_CKPT" "$ALLOC_CKPT" "$MESO_ARCHIVE" "$MESO_STATS" \
                "$SUBGRID" "$CATALOG" scripts/05_convert_bmd_dir.py \
                scripts/43_subset_prepared_imerg.py \
                scripts/44_coarsen_imerg_observations.py \
                scripts/72_v7_two_stage_osse.py scripts/76_rank_v7_imerg_ingestion.py \
                scripts/77_evaluate_v7_subgrid_resolution.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ -d "$BMD_DIR" ]] || { echo "ERROR: missing BMD directory $BMD_DIR" >&2; exit 1; }
[[ -e "$IMERG_SOURCE" ]] || compgen -G "$IMERG_SOURCE" >/dev/null || {
    echo "ERROR: no BMD-windowed IMERG archive matches $IMERG_SOURCE" >&2
    exit 1
}

next_day() {
    date -u -d "$1 +1 day" +%Y-%m-%d 2>/dev/null || \
        date -u -v+1d -j -f %Y-%m-%d "$1" +%Y-%m-%d
}

OBS_START="$(next_day "$START")"
OBS_END="$(next_day "$END")"
OBS_START_CLEAN="${OBS_START//-/}"
OBS_END_CLEAN="${OBS_END//-/}"
mkdir -p "$OUT/imerg"

NATIVE_IMERG="$OUT/imerg/imerg_0p1_${OBS_START_CLEAN}_${OBS_END_CLEAN}.nc"
S04_IMERG="$OUT/imerg/imerg_0p4_${OBS_START_CLEAN}_${OBS_END_CLEAN}.nc"
if [[ ! -s "$NATIVE_IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
        --input "$IMERG_SOURCE" --start "$OBS_START" --end "$OBS_END" \
        --out "$NATIVE_IMERG" --report "${NATIVE_IMERG%.nc}_qc.json"
fi
if [[ ! -s "$S04_IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/44_coarsen_imerg_observations.py \
        --input "$NATIVE_IMERG" --factor 8 --out "$S04_IMERG" \
        --report "${S04_IMERG%.nc}_qc.json"
fi

DAILY="$OUT/bmd_daily.csv"
if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DIR" --stations "$CATALOG" \
        --start "$START" --end "$OBS_END" \
        --out "$DAILY" --summary "$OUT/bmd_stations.csv" --report "$OUT/bmd_qc.json"
fi

V7_DUMP="$OUT/v7_station_ensembles.npz"
V7_MAP_DUMP="$OUT/v7_meanfields.npz"
echo "V7-only IMERG ingestion sweep"
echo "  model days: $START through $END; BMD days: $OBS_START through $OBS_END"
echo "  native: R x3/x9/x27/x81 plus R x27 ig010/L2"
echo "  S04: R x1 strong, correlation-adjusted robust, correlation-adjusted ig010/L2"
echo "  frozen checkpoints: $V7_REFERENCE_ROOT"
echo "  output: $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
    --meso-checkpoint "$MESO_CKPT" --allocation-checkpoint "$ALLOC_CKPT" \
    --meso-archive "$MESO_ARCHIVE" --meso-stats "$MESO_STATS" \
    --subgrid-archive "$SUBGRID" --stations "$DAILY" \
    --imerg "$NATIVE_IMERG" --imerg-s04 "$S04_IMERG" \
    --start "$START" --end "$END" \
    --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED" \
    --observations real --min-coverage 0.8 \
    --arms background,da_meso,da_sim --imerg-r-multiplier 9 \
    --imerg-r-sweep 3,27,81 --imerg-refine-r 27 \
    --station-dump "$V7_DUMP" --map-dump "$V7_MAP_DUMP" --out "$OUT"

RESULT="$OUT/v7_two_stage_real.json"
[[ -s "$RESULT" && -s "$V7_DUMP" && -s "$V7_MAP_DUMP" ]] || {
    echo "ERROR: V7 sweep did not write all expected outputs" >&2
    exit 1
}

"$PYTHON_BIN" -u scripts/76_rank_v7_imerg_ingestion.py \
    --input "$RESULT" \
    --out-json "$OUT/v7_imerg_ingestion_ranking.json" \
    --out-markdown "$OUT/v7_imerg_ingestion_ranking.md" \
    --out-plot "$OUT/v7_imerg_ingestion_ranking.png"

# Texture by itself is not resolution.  Strip each field's exact 0.1-degree
# block mean, test the remaining 2x2 placement against CHIRPS and a randomized
# within-cell null, and use withheld BMD anomalies as an independent point check.
"$PYTHON_BIN" -u scripts/77_evaluate_v7_subgrid_resolution.py \
    --map-dump "$V7_MAP_DUMP" --station-dump "$V7_DUMP" \
    --subgrid-archive "$SUBGRID" --arm da_sim_r81 \
    --out-dir "$OUT/subgrid_resolution"

echo "Winner report: $OUT/v7_imerg_ingestion_ranking.md"
echo "Subgrid evidence: $OUT/subgrid_resolution/v7_r81_subgrid_evidence.md"
