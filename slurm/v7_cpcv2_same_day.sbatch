#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-cpcv2
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# A matched 1--10 May model window, two winning DA modes, two model families.
#
# V7:
#   da_meso  gauges-only stage-A DA (current selected V7 defaults)
#   da_sim   simultaneous stage-A gauge + native 0.1-degree IMERG DA, R x9
#
# CPCv2:
#   guided_s6_g010_t100  frozen gauges-only winner
#   v2_simul_s04_ig010  frozen simultaneous S04 winner
#
# This is deliberately a REAL-observation test, not an OSSE: CPCv2's winning
# configurations were selected on BMD gauge verification.  The job locks V7's
# withheld IDs and feeds precisely that list to CPCv2; the final script refuses
# to score if the BMD values, dates, station pool, holdout IDs, or member count
# disagree. The model window is 1--10 May; BMD-windowed observations therefore
# run from 2--11 May.

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

START="${V7_COMPARE_START:-2022-05-01}"
END="${V7_COMPARE_END:-2022-05-10}"
[[ "$END" < "$START" ]] && {
    echo "ERROR: comparison end precedes start: START=$START END=$END" >&2
    exit 1
}
MEMBERS="${V7_COMPARE_MEMBERS:-16}"
STEPS="${V7_COMPARE_STEPS:-50}"
SEED="${V7_COMPARE_SEED:-20220501}"
OUT="${V7_COMPARE_OUT:-data/processed/v7_cpcv2_matched_may2022/$(date +%Y%m%d_%H%M)}"

MESO_CKPT="${V7_OSSE_MESO_CKPT:-runs/v7/meso/best.pt}"
ALLOC_CKPT="${V7_OSSE_ALLOC_CKPT:-runs/v7/allocation/best.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
# The authoritative BMD-windowed monthly archive made by
# download_imerg_halfhourly_2021_2024.sbatch.  Exact run-window products are
# made below so both DA systems use the same IMERG dates and raw source.
IMERG_SOURCE="${V7_COMPARE_IMERG_SOURCE:-data/processed/imerg_bd_aligned_*.nc}"
if [[ "$IMERG_SOURCE" == /path/to/* ]]; then
    echo "WARNING: ignoring placeholder IMERG source: $IMERG_SOURCE" >&2
    IMERG_SOURCE="data/processed/imerg_bd_aligned_*.nc"
fi

CPCV2_CKPT="${V7_COMPARE_CPCV2_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CPCV2_CONFIG="${V7_COMPARE_CPCV2_CONFIG:-configs/da.yaml}"
# CPCv2's frozen simultaneous winner uses S04 (factor 8 = 0.4 degrees), built
# from the exact native source that V7 uses at 0.1 degrees.

BMD_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

for required in "$MESO_CKPT" "$ALLOC_CKPT" "$MESO_ARCHIVE" "$MESO_STATS" \
                "$SUBGRID" "$CPCV2_CKPT" "$CPCV2_CONFIG" "$CATALOG" \
                scripts/05_convert_bmd_dir.py scripts/28_simultaneous_method_sweep.py \
                scripts/43_subset_prepared_imerg.py scripts/44_coarsen_imerg_observations.py \
                scripts/72_v7_two_stage_osse.py scripts/73_compare_v7_cpcv2_day.py \
                scripts/74_plot_v7_cpcv2_diagnostics.py; do
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

# BMD's date label is the 24 h ending 03 UTC on the following morning. V7's
# model days C therefore verify against BMD days C+1; CPCv2 represents this as
# observation days C+1 plus --background-day-offset -1.
CPCV2_START="$(next_day "$START")"
CPCV2_END="$(next_day "$END")"
CPCV2_START_CLEAN="${CPCV2_START//-/}"
CPCV2_END_CLEAN="${CPCV2_END//-/}"

# CPCv2 requires its source time axis to equal the requested checkpoint dates.
# Subset the archive to the BMD dates, then coarsen that exact file to S04.
mkdir -p "$OUT/checkpoints" "$OUT/imerg"
V7_IMERG="$OUT/imerg/imerg_aligned_${CPCV2_START_CLEAN}_${CPCV2_END_CLEAN}.nc"
CPCV2_IMERG="$OUT/imerg/imerg_0p4deg_${CPCV2_START_CLEAN}_${CPCV2_END_CLEAN}.nc"
if [[ ! -s "$V7_IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
        --input "$IMERG_SOURCE" \
        --start "$CPCV2_START" --end "$CPCV2_END" \
        --out "$V7_IMERG" --report "${V7_IMERG%.nc}_qc.json"
fi
if [[ ! -s "$CPCV2_IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/44_coarsen_imerg_observations.py \
        --input "$V7_IMERG" --factor 8 --out "$CPCV2_IMERG" \
        --report "${CPCV2_IMERG%.nc}_qc.json"
fi

DAILY="$OUT/bmd_daily.csv"
if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DIR" --stations "$CATALOG" \
        --start "$START" --end "$CPCV2_END" \
        --out "$DAILY" --summary "$OUT/bmd_stations.csv" --report "$OUT/bmd_qc.json"
fi

# CPCv2 does not snapshot a checkpoint itself. Freeze it before the two models
# are sampled so a late validation update cannot make this an epoch mismatch.
CPCV2_FROZEN="$OUT/checkpoints/cpcv2_frozen.pt"
cp -p "$CPCV2_CKPT" "$CPCV2_FROZEN"

V7_DUMP="$OUT/v7_station_ensembles.npz"
V7_MAP_DUMP="$OUT/v7_meanfields.npz"
CPCV2_DUMP="$OUT/cpcv2_station_ensembles.npz"
WITHHELD_IDS="$OUT/v7_withheld_station_ids.txt"

echo "Matched V7 vs CPCv2 comparison"
echo "  model days: $START through $END; BMD observation days: $CPCV2_START through $CPCV2_END"
echo "  V7 arms: background (diagnostic), da_meso, da_sim (IMERG R x9)"
echo "  CPCv2 arms: guided_s6_g010_t100, v2_simul_s04_ig010"
echo "  sampling: $MEMBERS members, $STEPS Heun steps, seed $SEED"
echo "  output: $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
    --meso-checkpoint "$MESO_CKPT" \
    --allocation-checkpoint "$ALLOC_CKPT" \
    --meso-archive "$MESO_ARCHIVE" \
    --meso-stats "$MESO_STATS" \
    --subgrid-archive "$SUBGRID" \
    --stations "$DAILY" \
    --imerg "$V7_IMERG" \
    --start "$START" --end "$END" \
    --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED" \
    --observations real --min-coverage 0.8 \
    --arms background,da_meso,da_sim --imerg-r-multiplier 9 \
    --station-dump "$V7_DUMP" --map-dump "$V7_MAP_DUMP" \
    --out "$OUT"

[[ -s "$WITHHELD_IDS" && -s "$V7_DUMP" && -s "$V7_MAP_DUMP" ]] || {
    echo "ERROR: V7 did not write the locked split, station dump, and map dump" >&2
    exit 1
}

# Exercise the same sqrt-space guidance path CPCv2 will use before comparing.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py
srun --ntasks=1 "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CPCV2_CONFIG" --ckpt "$CPCV2_FROZEN" \
    --stations "$DAILY" --imerg "$CPCV2_IMERG" \
    --start "$CPCV2_START" --end "$CPCV2_END" \
    --background-day-offset -1 \
    --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED" \
    --min-coverage 0.8 --holdout-station-ids-file "$WITHHELD_IDS" \
    --group v2_comparison \
    --set observations.imerg.factor=8 \
    --set observations.imerg.error_corr_cells=0.75 \
    --out "$CPCV2_DUMP" --report "$OUT/cpcv2_report.json"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/73_compare_v7_cpcv2_day.py \
    --v7-dump "$V7_DUMP" --cpcv2-dump "$CPCV2_DUMP" \
    --out-json "$OUT/v7_cpcv2_matched_comparison.json" \
    --out-markdown "$OUT/v7_cpcv2_matched_comparison.md"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/74_plot_v7_cpcv2_diagnostics.py \
    --v7-dump "$V7_DUMP" --cpcv2-dump "$CPCV2_DUMP" \
    --v7-map-dump "$V7_MAP_DUMP" --out-dir "$OUT/diagnostics"

echo "Comparison complete: $OUT/v7_cpcv2_matched_comparison.md"
echo "DA diagnostics: $OUT/diagnostics"
