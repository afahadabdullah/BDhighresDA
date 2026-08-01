#!/bin/bash
#SBATCH --job-name=bdhires-bmd-example
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Real-gauge process gate: convert the legacy BMD archive, assimilate 24 of the
# 30 available May 2018 stations, and evaluate only at six geographically
# spread withheld stations. IMERG is deliberately excluded from this first
# real-data test so BMD ingestion/guidance can be diagnosed in isolation.

set -euo pipefail

module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

RAINFALL="${BMD_RAINFALL:-data/bmd/bmd.csv}"
STATIONS="${BMD_STATIONS:-data/bmd/Stations.csv}"
START="${BMD_START:-2018-05-01}"
END="${BMD_END:-2018-05-31}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
MEMBERS="${BMD_MEMBERS:-16}"
WITHHOLD="${BMD_WITHHOLD:-0.2}"

DAILY="${BMD_DAILY:-data/processed/bmd_daily_may2018.csv}"
STATION_SUMMARY="${BMD_STATION_SUMMARY:-data/processed/bmd_stations_may2018.csv}"
QC_REPORT="${BMD_QC_REPORT:-data/processed/bmd_qc_may2018.json}"
DUMP="${BMD_DUMP:-data/processed/bmd_may2018_example.npz}"
REPORT="${BMD_REPORT:-data/processed/bmd_may2018_example.json}"
DIAGNOSTICS="${BMD_DIAGNOSTICS:-data/processed/bmd_may2018_diagnostics.png}"
SPATIAL="${BMD_SPATIAL:-data/processed/bmd_may2018_spatial.png}"

for required in "$RAINFALL" "$STATIONS" "$CKPT" "$CONFIG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done

echo "BMD rainfall: $RAINFALL"
echo "BMD stations: $STATIONS"
echo "Period: $START to $END"
echo "Checkpoint: $CKPT"
echo "Members: $MEMBERS   Withhold: $WITHHOLD"
echo "NOTE: 2018 is in the checkpoint training period; this is a workflow test, not final skill."
echo "NOTE: This first real-data gate assimilates BMD gauges only, not IMERG."

srun --ntasks=1 "$PYTHON_BIN" -u scripts/05_convert_bmd_legacy.py \
    --rainfall "$RAINFALL" \
    --stations "$STATIONS" \
    --start "$START" \
    --end "$END" \
    --out "$DAILY" \
    --summary "$STATION_SUMMARY" \
    --report "$QC_REPORT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/15_bmd_month_example.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --stations "$DAILY" \
    --start "$START" \
    --end "$END" \
    --members "$MEMBERS" \
    --withhold "$WITHHOLD" \
    --out "$DUMP" \
    --report "$REPORT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/16_plot_bmd_month_example.py \
    --dump "$DUMP" \
    --report "$REPORT" \
    --out-diagnostics "$DIAGNOSTICS" \
    --out-spatial "$SPATIAL"

echo "BMD May 2018 example complete."
echo "Metrics: $REPORT"
echo "Diagnostics: $DIAGNOSTICS"
echo "Spatial cases: $SPATIAL"
