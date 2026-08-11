#!/bin/bash
#SBATCH --job-name=bdhires-osse-day
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#
# One day, several configurations, one comparison figure.
#
# A 48-day arm is the wrong instrument for "does this idea help at all". This
# runs a single nature-run day through every variant in sequence, so the whole
# sweep costs about what one 48-day arm costs per variant divided by 48, and
# the answer arrives in minutes rather than hours.
#
# DEFAULT DAY 2023-03-19 is chosen because it is where the failure is visible:
# domain mean 30.5 mm/day, large organised structures in the truth, a nearly
# featureless background, and an analysis that put bright spots on the gauges
# and left everything between them alone.
#
# The variants sweep the three levers that could plausibly fix that, and they
# are NOT equivalent:
#   spread   gives the increment spatial extent (new; see GuidanceConfig)
#   gamma    scales how hard the likelihood pulls
#   temp     widens the prior ensemble, so there is more structure to correct
# Only the first addresses a nearly diagonal implied covariance directly; the
# other two are included because they are cheap here and because if one of them
# fixes it, the diagnosis was wrong.
#
# Usage
#   sbatch slurm/osse_single_day.sbatch
#   DAY=2023-09-22 sbatch slurm/osse_single_day.sbatch
#   VARIANTS="base|0|||  s12|12|||" sbatch slurm/osse_single_day.sbatch

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
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: needs an aarch64 Grace node (got $(uname -m))."; exit 1; }

# srun inside an allocation fails on these nodes; see bmd_imerg_example.sbatch.
read -r -a LAUNCH <<< "${BDHIRES_LAUNCH:-}"
run_step() { "${LAUNCH[@]+"${LAUNCH[@]}"}" "$@"; }

# Attribute failures to hardware. A CUDA "invalid access of peer GPU memory
# over nvlink or a hardware error" is a driver/device fault, not a Python bug,
# and without the node name and device state in the log there is no way to tell
# a bad node from a real regression on the next run.
echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
    --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
# Reduces fragmentation on long multi-variant runs; harmless otherwise.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Set CUDA_LAUNCH_BLOCKING=1 to make an async CUDA fault report at its true
# call site. It costs speed, so it is opt-in.
[[ -n "${CUDA_LAUNCH_BLOCKING:-}" ]] && echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

DAY="${DAY:-2023-03-19}"
CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
MEMBERS="${DAY_MEMBERS:-50}"
WITHHOLD="${DAY_WITHHOLD:-0.2}"
SEED="${DAY_SEED:-7}"
NETWORK="${DAY_NETWORK:-bmd}"
HOLDOUT="${DAY_HOLDOUT:-neighbored}"
NEIGHBOR_KM="${DAY_NEIGHBOR_KM:-75}"
MAX_GAP_DEG="${DAY_MAX_GAP_DEG:-200}"
CATALOG="${OSSE_BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
GRID="${DAY_GRID:-128}"          # fine-grid side length, for the crop arithmetic
ROOT="${DAY_ROOT:-data/processed/osse_day_${DAY//-/}}"
[[ -s "$CATALOG" ]] || { echo "ERROR: no catalogue at $CATALOG"; exit 1; }

# NAME|SPREAD_CELLS|GAMMA|PRIOR_TEMPERATURE|MODE|SAT_SOURCE|SAT_FACTOR
# Empty fields leave the config alone. MODE is gauges|satellite|combined,
# SAT_SOURCE is truth (CHIRPS block means) or cpc, SAT_FACTOR is in 0.05-degree
# cells so 10 = 0.5 degrees, the CPC resolution.
VARIANTS="${VARIANTS:-
base|0||
s6|6||
s12|12||
s20|20||
g1e-2|0|1.0e-2|
temp15|0||1.5
s12temp15|12||1.5
}"

echo "============================================================"
echo " Single-day OSSE sweep"
echo "   day        $DAY"
echo "   checkpoint $CKPT"
echo "   network    $NETWORK, holdout $HOLDOUT (<=${NEIGHBOR_KM} km, <=${MAX_GAP_DEG} deg)"
echo "   members    $MEMBERS"
echo "   out        $ROOT/"
echo "============================================================"

for spec in $VARIANTS; do
    [[ -z "$spec" ]] && continue
    IFS='|' read -r NAME SPREAD GAMMA TEMP MODE SATSRC SATFAC <<< "$spec"
    MODE="${MODE:-gauges}"; SATSRC="${SATSRC:-truth}"; SATFAC="${SATFAC:-10}"
    OUT="$ROOT/$NAME"
    if [[ -s "$OUT/ensemble.npz" ]]; then
        echo ">>> $NAME already present, skipping"; continue
    fi
    # One variant failing must not lose the others: each is an independent
    # question, and a device fault on one is not evidence about the rest.
    FAILED=0
    mkdir -p "$OUT"
    echo
    echo ">>> variant $NAME  mode=$MODE spread=${SPREAD:-0} gamma=${GAMMA:-config} temp=${TEMP:-config}"
    [[ "$MODE" != "gauges" ]] && echo "    satellite: source=$SATSRC factor=$SATFAC (= $(awk -v f=$SATFAC 'BEGIN{printf "%.2f", f*0.05}') deg)"

    EXTRA=()
    [[ -n "${SPREAD:-}" ]] && EXTRA+=(--guidance-spread-cells "$SPREAD")
    [[ -n "${GAMMA:-}"  ]] && EXTRA+=(--guidance-gamma "$GAMMA")
    [[ -n "${TEMP:-}"   ]] && EXTRA+=(--prior-temperature "$TEMP")
    if [[ "$MODE" != "gauges" ]]; then
        # stride 1 and no correlation control: the observation is exact, so
        # there is no correlated error to guard against and inflating R would
        # only down-weight perfect information.
        EXTRA+=(--satellite-factor "$SATFAC" --satellite-stride 1
                --satellite-source "$SATSRC")
        [[ -n "${SATSIG:-}" ]] && EXTRA+=(--satellite-sigma "$SATSIG")
        # A factor that does not divide the grid needs an explicit window.
        # 128 / 10 = 12.8, so the 0.5-degree CPC footprint requires cropping to
        # 120 and gives up the northern and eastern 0.4-degree strip. The crop
        # starts at 0 so the footprints tile from the grid origin; an offset
        # window would nest differently from every other factor.
        CROPSZ=$(( (GRID / SATFAC) * SATFAC ))
        if [[ "$CROPSZ" -ne "$GRID" ]]; then
            echo "    crop: ${GRID}x${GRID} is not divisible by $SATFAC; using 0..$CROPSZ"
            echo "          (drops the outer $(( GRID - CROPSZ )) cells = $(awk -v n=$(( GRID - CROPSZ )) 'BEGIN{printf "%.2f", n*0.05}') deg on N and E)"
            EXTRA+=(--satellite-crop 0 "$CROPSZ" 0 "$CROPSZ")
        fi
    fi

    run_step "$PYTHON_BIN" -u scripts/10_osse.py \
        --config "$CONFIG" --ckpt "$CKPT" \
        --start "$DAY" --end "$DAY" --month 0 --days 1 \
        --members "$MEMBERS" \
        --networks "$NETWORK" --bmd-stations "$CATALOG" \
        --station-layout spread \
        --holdout-layout "$HOLDOUT" \
        --holdout-neighbor-km "$NEIGHBOR_KM" \
        --holdout-max-gap-deg "$MAX_GAP_DEG" \
        --withhold "$WITHHOLD" --seed "$SEED" \
        --obs-error exact --observation-mode "$MODE" \
        --dump "$OUT/ensemble.npz" \
        --dump-network "$NETWORK" --dump-obs-error exact \
        --out-figure "$OUT/osse_overview.png" \
        --out-report "$OUT/osse_report.json" \
        "${EXTRA[@]+"${EXTRA[@]}"}" || FAILED=1
    if [[ "$FAILED" == "1" ]]; then
        echo "!!! variant $NAME FAILED on $(hostname); continuing with the rest." >&2
        echo "    If this is a CUDA device fault, retry elsewhere:" >&2
        echo "      sbatch --exclude=$(hostname) slurm/osse_single_day.sbatch" >&2
        rm -f "$OUT/ensemble.npz"
    fi
done

echo
echo ">>> comparison figure"
run_step "$PYTHON_BIN" -u scripts/48_single_day_compare.py \
    --root "$ROOT" --day "$DAY" --out-dir "$ROOT"

echo
echo "============================================================"
echo " $ROOT/single_day_compare.png   maps, one row per variant"
echo " $ROOT/single_day_compare.csv   the numbers"
echo "============================================================"
