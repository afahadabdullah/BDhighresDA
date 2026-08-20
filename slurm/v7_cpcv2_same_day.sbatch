#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-cpcv2
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# One BMD-aligned observation day, two winning DA modes, two model families.
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
# disagree.

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

START="${V7_COMPARE_START:-2022-05-03}"
END="${V7_COMPARE_END:-$START}"
[[ "$START" == "$END" ]] || {
    echo "ERROR: this is intentionally a same-day comparison; START=$START END=$END" >&2
    exit 1
}
MEMBERS="${V7_COMPARE_MEMBERS:-16}"
STEPS="${V7_COMPARE_STEPS:-50}"
SEED="${V7_COMPARE_SEED:-20220503}"
OUT="${V7_COMPARE_OUT:-data/processed/v7_cpcv2_same_day/$(date +%Y%m%d_%H%M)}"

MESO_CKPT="${V7_OSSE_MESO_CKPT:-runs/v7/meso/best.pt}"
ALLOC_CKPT="${V7_OSSE_ALLOC_CKPT:-runs/v7/allocation/best.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
# Reuse the paired 1--10 May products prepared by
# submit_v2_ingestion_triplet.sh: native 0.1-degree for V7 and factor-8/S04 for
# CPCv2.  The default model day (03 May) and its BMD label day (04 May) are both
# covered.  Overrides remain available for an intentionally different window.
IMERG_PREP_DIR="${V7_COMPARE_IMERG_DIR:-${V2_INGEST_IMERG_DIR:-data/processed/imerg_prepared_ing2022}}"
V7_IMERG="${V7_COMPARE_V7_IMERG:-${V7_OSSE_IMERG:-}}"
if [[ "$V7_IMERG" == /path/to/* ]]; then
    echo "WARNING: ignoring placeholder V7 IMERG path: $V7_IMERG" >&2
    V7_IMERG=""
fi
V7_IMERG="${V7_IMERG:-$IMERG_PREP_DIR/imerg_aligned_20220501_20220510.nc}"

CPCV2_CKPT="${V7_COMPARE_CPCV2_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CPCV2_CONFIG="${V7_COMPARE_CPCV2_CONFIG:-configs/da.yaml}"
# CPCv2's frozen simultaneous winner used S04 (factor 8 = 0.4 degrees), not
# V7's native 0.1-degree stream.  Keep each model's winning observation setup.
CPCV2_IMERG="${V7_COMPARE_CPCV2_IMERG:-}"
if [[ "$CPCV2_IMERG" == /path/to/* ]]; then
    echo "WARNING: ignoring placeholder CPCv2 IMERG path: $CPCV2_IMERG" >&2
    CPCV2_IMERG=""
fi
CPCV2_IMERG="${CPCV2_IMERG:-$IMERG_PREP_DIR/imerg_0p4deg_20220501_20220510.nc}"

BMD_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

for required in "$MESO_CKPT" "$ALLOC_CKPT" "$MESO_ARCHIVE" "$MESO_STATS" \
                "$SUBGRID" "$V7_IMERG" "$CPCV2_CKPT" "$CPCV2_CONFIG" \
                "$CPCV2_IMERG" "$CATALOG" \
                scripts/05_convert_bmd_dir.py scripts/28_simultaneous_method_sweep.py \
                scripts/72_v7_two_stage_osse.py scripts/73_compare_v7_cpcv2_day.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ -d "$BMD_DIR" ]] || { echo "ERROR: missing BMD directory $BMD_DIR" >&2; exit 1; }

next_day() {
    date -u -d "$1 +1 day" +%Y-%m-%d 2>/dev/null || \
        date -u -v+1d -j -f %Y-%m-%d "$1" +%Y-%m-%d
}

# BMD's date label is the 24 h ending 03 UTC on the following morning.  V7's
# model day C therefore verifies against BMD day C+1; CPCv2 represents this as
# observation day C+1 plus --background-day-offset -1.
CPCV2_DAY="$(next_day "$START")"

mkdir -p "$OUT/checkpoints"
DAILY="$OUT/bmd_daily.csv"
if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DIR" --stations "$CATALOG" \
        --start "$START" --end "$CPCV2_DAY" \
        --out "$DAILY" --summary "$OUT/bmd_stations.csv" --report "$OUT/bmd_qc.json"
fi

# CPCv2 does not snapshot a checkpoint itself. Freeze it before the two models
# are sampled so a late validation update cannot make this an epoch mismatch.
CPCV2_FROZEN="$OUT/checkpoints/cpcv2_frozen.pt"
cp -p "$CPCV2_CKPT" "$CPCV2_FROZEN"

V7_DUMP="$OUT/v7_station_ensembles.npz"
CPCV2_DUMP="$OUT/cpcv2_station_ensembles.npz"
WITHHELD_IDS="$OUT/v7_withheld_station_ids.txt"

echo "Matched V7 vs CPCv2 one-day comparison"
echo "  model day: $START; BMD observation day: $CPCV2_DAY"
echo "  V7 arms: da_meso, da_sim (IMERG R x9)"
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
    --arms da_meso,da_sim --imerg-r-multiplier 9 \
    --station-dump "$V7_DUMP" \
    --out "$OUT"

[[ -s "$WITHHELD_IDS" && -s "$V7_DUMP" ]] || {
    echo "ERROR: V7 did not write the locked split and station dump" >&2
    exit 1
}

# Exercise the same sqrt-space guidance path CPCv2 will use before comparing.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py
srun --ntasks=1 "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CPCV2_CONFIG" --ckpt "$CPCV2_FROZEN" \
    --stations "$DAILY" --imerg "$CPCV2_IMERG" \
    --start "$CPCV2_DAY" --end "$CPCV2_DAY" \
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

echo "Comparison complete: $OUT/v7_cpcv2_matched_comparison.md"
