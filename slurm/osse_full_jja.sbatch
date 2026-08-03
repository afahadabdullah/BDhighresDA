#!/bin/bash
#SBATCH --job-name=bdhires-osse-full
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-35%6
#SBATCH --output=logs/%x-%A_%a.out

# Maximum primary paper OSSE: all 368 JJA days in 2021--2024, 16 members and
# three matched exact-observation arms. Each array task handles one arm/month.

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
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
MEMBERS="${OSSE_FULL_MEMBERS:-16}"
WITHHOLD="${OSSE_FULL_WITHHOLD:-0.2}"
SEED="${OSSE_FULL_SEED:-7}"
ROOT="${OSSE_FULL_ROOT:-data/processed/osse_full_jja_2021_2024}"
NETWORK="${OSSE_FULL_NETWORK:-bmd}"
NETWORK_TAG="${OSSE_FULL_NETWORK_TAG:-bmd}"
BMD_STATIONS="${OSSE_BMD_STATIONS:-data/bmd/Stations.csv}"
if [[ ! -f "$BMD_STATIONS" && -f data/stations/Stations.csv ]]; then
    BMD_STATIONS="data/stations/Stations.csv"
fi

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
if [[ "${NETWORK,,}" == "bmd" ]]; then
    [[ -f "$BMD_STATIONS" ]] || {
        echo "ERROR: BMD station catalogue not found: $BMD_STATIONS"
        exit 1
    }
fi

ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 12))
GROUP_INDEX=$((SLURM_ARRAY_TASK_ID % 12))
YEAR=$((2021 + GROUP_INDEX / 3))
MONTH=$((6 + GROUP_INDEX % 3))

case "$ARM_INDEX" in
    0) LABEL="gauges_exact_${NETWORK_TAG}"; MODE="gauges" ;;
    1) LABEL="satellite_exact_${NETWORK_TAG}"; MODE="satellite" ;;
    2) LABEL="simultaneous_exact_${NETWORK_TAG}"; MODE="combined" ;;
    *) echo "ERROR: unknown arm index $ARM_INDEX"; exit 2 ;;
esac

CHUNK="${YEAR}_$(printf '%02d' "$MONTH")"
OUT="$ROOT/chunks/$LABEL/$CHUNK"
mkdir -p "$OUT"

echo "Full OSSE chunk: $LABEL / $YEAR-$(printf '%02d' "$MONTH")"
echo "mode=$MODE; network=$NETWORK; exact observations; members=$MEMBERS; seed=$SEED"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$YEAR-06-01" \
    --end "$YEAR-08-31" \
    --month "$MONTH" \
    --days 31 \
    --members "$MEMBERS" \
    --networks "$NETWORK" \
    --bmd-stations "$BMD_STATIONS" \
    --station-layout spread \
    --holdout-layout spread \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error exact \
    --observation-mode "$MODE" \
    --satellite-stride 1 \
    --dump "$OUT/ensemble.npz" \
    --dump-network "$NETWORK" \
    --dump-obs-error exact \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json"

[[ -s "$OUT/ensemble.npz" ]] || {
    echo "ERROR: missing ensemble chunk $OUT/ensemble.npz"
    exit 3
}

echo "completed $LABEL / $CHUNK"
