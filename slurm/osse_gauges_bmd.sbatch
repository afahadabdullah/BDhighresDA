#!/bin/bash
#SBATCH --job-name=bdhires-osse-gauges
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out

# Gauges-only OSSE on the real BMD geometry. No satellite anywhere.
#
# CHIRPS over 2021-2024 is the 0.05-degree nature run and supplies pseudo-gauge
# values at the actual BMD station coordinates, so the OSSE and the real-data
# experiment share a network geometry and can be read against each other
# directly. Everything is scored on WITHHELD pseudo-gauges.
#
# Withhold 0.2 of ~40 stations is 8 withheld against 32 assimilated, matching
# the real-data folds (8 of 38). Matching it is the point: a difference between
# the OSSE and the real experiment should not be attributable to how much was
# held out.
#
# WHY 2021-2024 AND NOT 2020: the prior splits train [1981,2018],
# val [2019,2020], test [2021,2025]. Scoring on 2020 would use years the
# checkpoint selection saw.
#
# The three tasks:
#   0  gauges_exact_bmd      exact pseudo-observations. The mechanistic upper
#                            bound -- what the DA can do when the observation
#                            is the truth and only R's numerical floor remains.
#   1  gauges_realistic_bmd  observation error from configs/da.yaml, i.e. the
#                            operational setting.
#   2  gauges_tune_bmd       the configuration matrix: gamma x sigma_obs x
#                            prior_temperature (3 x 4 x 5 = 60), scored on the
#                            same withheld pseudo-gauges.
#
# Task 2 is the one that interrogates the real-data MEASR paradox. There, using
# the MEASURED representativeness (0.410) made things WORSE, and the proposed
# explanation was the prior's +8.37 mm/day wet bias rather than a bad
# measurement. The tuning grid includes sigma_obs = 0.40, so if 0.40 is near
# optimal against a known truth while it was worst against real gauges, the
# explanation is supported. If 0.40 is bad here too, it is not.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
DAYS="${OSSE_GAUGE_DAYS:-48}"
MEMBERS="${OSSE_GAUGE_MEMBERS:-16}"
START="${OSSE_GAUGE_START:-2021-01-01}"
END="${OSSE_GAUGE_END:-2024-12-31}"
WITHHOLD="${OSSE_GAUGE_WITHHOLD:-0.2}"
SEED="${OSSE_GAUGE_SEED:-7}"
ROOT="${OSSE_GAUGE_ROOT:-data/processed/osse_gauges}"
NETWORK="${OSSE_GAUGE_NETWORK:-bmd}"
TUNE_DAYS="${OSSE_GAUGE_TUNE_DAYS:-16}"

# The 'bmd' network token reads coordinates from this catalogue. The default in
# scripts/10_osse.py points at data/bmd/Stations.csv, which does not exist in
# this checkout; the per-station archive lives here instead.
CATALOG="${OSSE_BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
[[ -s "$CATALOG" ]] || {
    echo "ERROR: BMD station catalogue not found: $CATALOG"
    echo "  Set OSSE_BMD_STATIONS to the Stations.csv with lat/lon columns."
    exit 1
}

case "$SLURM_ARRAY_TASK_ID" in
    0) LABEL="gauges_exact_bmd";     ERROR="exact";     TUNE="0" ;;
    1) LABEL="gauges_realistic_bmd"; ERROR="realistic"; TUNE="0" ;;
    2) LABEL="gauges_tune_bmd";      ERROR="realistic"; TUNE="1" ;;
    *) echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

OUT="$ROOT/$LABEL"
mkdir -p "$OUT"

echo "Gauges-only OSSE arm: $LABEL"
echo "network=$NETWORK (catalogue $CATALOG) error=$ERROR tune=$TUNE"
echo "window=$START..$END days=$DAYS members=$MEMBERS withhold=$WITHHOLD seed=$SEED"
echo "NO satellite in any arm."

EXTRA=()
if [[ "$TUNE" == "1" ]]; then
    # --tune replaces the network sweep with a configuration sweep and forces
    # the 'realistic' error level. --tune-network is REQUIRED here: without it
    # the script picks the largest SYNTHETIC network and would silently ignore
    # the BMD geometry this experiment is about.
    EXTRA+=(--tune --tune-network "$NETWORK" --tune-days "$TUNE_DAYS")
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$START" \
    --end "$END" \
    --month 0 \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --networks "$NETWORK" \
    --bmd-stations "$CATALOG" \
    --station-layout spread \
    --holdout-layout spread \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error "$ERROR" \
    --observation-mode gauges \
    --dump "$OUT/ensemble.npz" \
    --dump-network "$NETWORK" \
    --dump-obs-error "$ERROR" \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

if [[ "$TUNE" == "1" ]]; then
    echo "Tuning arm complete: $OUT/osse_report.json holds the 60-combination"
    echo "matrix. The downscaling claims are not rebuilt for it -- they describe"
    echo "one configuration, and this arm deliberately has sixty."
    exit 0
fi

# Same downstream diagnostics as the satellite paper arms, so figures 4-7 and
# Table 1 read these directories without special-casing.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
    --dump "$OUT/ensemble.npz" \
    --out-figure "$OUT/da_diagnostics.png" \
    --out-report "$OUT/da_diagnostics.json"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/14_osse_summary.py \
    --dump "$OUT/ensemble.npz" \
    --cases 4 \
    --out-reconstruction "$OUT/reconstruction.png" \
    --out-matrix "$OUT/scale_matrix.png" \
    --out-report "$OUT/scale_summary.json"

# Claim A (downscaling gain over the coarse input) and Claim B (sub-footprint
# gain over a null handed perfect footprint means). Claim B is still meaningful
# with no satellite assimilated -- in fact it is cleaner, because no observation
# resolves sub-footprint structure and any skill there is the prior's alone.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
    --dump "$OUT/ensemble.npz" \
    --label "$LABEL" \
    --minimum-valid-fraction 1.0 \
    --out-figure "$OUT/downscaling_diagnostics.png" \
    --out-report "$OUT/downscaling.json" \
    --out-curve-data "$OUT/downscaling_curves.npz" \
    --out-spatial-data "$OUT/spatial_fields.nc"

echo "Arm $LABEL complete: $OUT"
