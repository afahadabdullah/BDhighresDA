#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-gauge-weight
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# One-fold, ten-day CPC-v2 gauge-authority screen.  Every arm uses the frozen
# production checkpoint, 30 members, the production seed, the same held-out
# stations, and the exact S04 IMERG archive used by the 2021-2024 production
# run.  Only gauge likelihood weight changes.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-gauge-weight-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

START="${V2_GW_START:-2022-05-01}"
END="${V2_GW_END:-2022-05-10}"
MEMBERS="${V2_GW_MEMBERS:-30}"
FOLD="${V2_GW_FOLD:-0}"
FOLDS="${V2_GW_FOLDS:-5}"
SEED="${V2_GW_SEED:-201805}"
ROOT="${V2_GW_ROOT:-data/processed/v2_gauge_authority_may2022}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
IMERG_SOURCE="${V2_GW_IMERG_SOURCE:-data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc}"
IMERG="${V2_GW_IMERG:-$ROOT/imerg_s04_20220501_20220510.nc}"

PREFIX="$ROOT/fold${FOLD}"
DAILY="${PREFIX}_bmd.csv"
STATION_SUMMARY="${PREFIX}_stations.csv"
QC="${PREFIX}_bmd_qc.json"
mkdir -p "$ROOT"

for required in "$CKPT" "$CONFIG" "$STATION_CATALOG" "$IMERG_SOURCE"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "ERROR: BMD data directory missing: $DATA_DIR"; exit 1; }

if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" \
      && -s "${PREFIX}_matrix.json" && -s "${PREFIX}_matrix.md" ]]; then
    echo "Gauge-authority screen already complete; reusing $PREFIX"
    exit 0
fi
if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
    echo "ERROR: partial DA output at $PREFIX; inspect or move it aside before retrying" >&2
    exit 1
fi

if [[ ! -s "$IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
        --input "$IMERG_SOURCE" --start "$START" --end "$END" \
        --out "$IMERG" --report "${IMERG%.nc}_qc.json"
else
    echo "Reusing exact ten-day S04 IMERG: $IMERG"
fi

if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$DATA_DIR" --stations "$STATION_CATALOG" \
        --start "$START" --end "$END" \
        --out "$DAILY" --summary "$STATION_SUMMARY" --report "$QC"
else
    echo "Reusing $DAILY"
fi

"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

echo "CPC-v2 gauge-authority screen"
echo "  period: $START through $END; fixed holdout fold $((FOLD + 1))/$FOLDS"
echo "  members: $MEMBERS; seed: $SEED; checkpoint: $CKPT"
echo "  arms: background; gauge-only and simultaneous weights 0.75,1,1.25,1.5,2,4,100-stress; IMERG-only"
echo "  IMERG: exact production S04 source $IMERG_SOURCE"
echo "  output: $PREFIX"

"$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --stations "$DAILY" --imerg "$IMERG" \
    --start "$START" --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" \
    --holdout-folds "$FOLDS" --holdout-fold "$FOLD" \
    --group v2_gauge_authority \
    --set observations.imerg.factor=8 \
    --set observations.imerg.error_corr_cells=0.75 \
    --seed "$SEED" \
    --out "${PREFIX}.npz" --report "${PREFIX}.json"

"$PYTHON_BIN" -u scripts/29_summarize_method_sweep.py \
    --dump "${PREFIX}.npz" --report "${PREFIX}.json" \
    --baseline v2_simul_s04_ig010_gw1 \
    --block-days 3 --n-resamples "${V2_GW_BOOTSTRAPS:-10000}" --seed "$SEED" \
    --out-json "${PREFIX}_matrix.json" \
    --out-markdown "${PREFIX}_matrix.md" \
    --out-plot "${PREFIX}_matrix.png"

echo "Gauge-authority screen complete."
echo "  table:  ${PREFIX}_matrix.md"
echo "  data:   ${PREFIX}_matrix.json"
echo "  figure: ${PREFIX}_matrix.png"
