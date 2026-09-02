#!/usr/bin/env bash
#SBATCH --job-name=bdhires-dense-sweep
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out

# One profile of the dense BMD+BWDB gauge DA sweep.
#
# A profile is a config the whole method group runs under; the arms inside it
# are the methods.  Splitting the experiment this way is not cosmetic: an arm
# is a Variant field and can change per method, while sigma/representativeness
# and the guidance norm cap are config-level and cannot.  Every arm inside one
# profile shares the background draw, the day seeds, the observation
# perturbations and the withheld stations, so a within-profile difference is
# the method; profiles differ only by the --set on this line, so a
# between-profile difference at the same arm is that override.
#
#   0 base           frozen production config
#   1 repr_measured  gauge representativeness set to the MEASURED value
#   2 repr_half      halfway between the production guess and the measurement
#   3 clip2000       guidance.clip_norm raised 10x
#   4 superob        gauges aggregated to one super-observation per cell
#   5 ensrf          localized serial EnSRF instead of joint guidance

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
mkdir -p logs
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: Grace/GH200 aarch64 node required"; exit 1; }
"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

ROOT="${V2_DENSE_ROOT:-data/processed/v2_dense_gauge_sweep}"
STATION_ROOT="$ROOT/stations"
START="${V2_DENSE_START:-2022-05-01}"
END="${V2_DENSE_END:-2022-05-05}"
MEMBERS="${V2_DENSE_MEMBERS:-30}"
SEED="${V2_DENSE_SEED:-202205}"
CELL_DEG="${V2_DENSE_CELL_DEG:-0.25}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
IMERG_SOURCE="${V2_DENSE_IMERG:-data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc}"
IMERG="${V2_DENSE_IMERG_SUBSET:-$ROOT/imerg_s04_${START//-/}_${END//-/}.nc}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
GROUP_JOINT="${V2_DENSE_GROUP:-v2_dense_gauge}"
GROUP_ENSRF="${V2_DENSE_GROUP_ENSRF:-v2_dense_gauge_ensrf}"

LABELS=(base repr_measured repr_half clip2000 superob ensrf)
TASK_ID="${SLURM_ARRAY_TASK_ID:?configured as 0..5}"
LABEL="${LABELS[$TASK_ID]}"

STATIONS="$STATION_ROOT/combined_eligible.csv"
HOLDOUT="$STATION_ROOT/holdout/fold0.txt"
BUDGET="$STATION_ROOT/error_budget.json"
SUPEROB_REPORT="$STATION_ROOT/superob_${CELL_DEG}.json"
GROUP="$GROUP_JOINT"
EXTRA_SET=()

read_budget () {
    # $1 = json file, $2 = key under recommended_representativeness
    "$PYTHON_BIN" -c "
import json, sys
payload = json.load(open(sys.argv[1]))['recommended_representativeness']
if payload is None:
    raise SystemExit(
        'ERROR: ' + sys.argv[1] + ' has no measured error budget. The prepare '
        'job needs --stats pointing at a real data/processed/stats.json; '
        'without it the transformed-space budget cannot be computed and the '
        'profile would silently run the production value.'
    )
print(payload[sys.argv[2]])
" "$1" "$2"
}

case "$LABEL" in
    base) ;;
    repr_measured)
        VALUE="$(read_budget "$BUDGET" implied_representativeness)"
        EXTRA_SET+=(--set "observations.gauges.representativeness=$VALUE")
        ;;
    repr_half)
        VALUE="$("$PYTHON_BIN" -c "
import json
payload = json.load(open('$BUDGET'))['recommended_representativeness']
print(round(0.5 * (0.25 + payload['implied_representativeness']), 4))
")"
        EXTRA_SET+=(--set "observations.gauges.representativeness=$VALUE")
        ;;
    clip2000)
        EXTRA_SET+=(--set "guidance.clip_norm=2000.0")
        ;;
    superob)
        STATIONS="$STATION_ROOT/superob_${CELL_DEG}.csv"
        VALUE="$(read_budget "$SUPEROB_REPORT" superob_implied_representativeness)"
        EXTRA_SET+=(--set "observations.gauges.representativeness=$VALUE")
        ;;
    ensrf)
        GROUP="$GROUP_ENSRF"
        ;;
    *) echo "ERROR: unknown profile $LABEL" >&2; exit 1 ;;
esac

PREFIX="$ROOT/profiles/$LABEL"
mkdir -p "$ROOT/profiles"
if [[ ! -s "$IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
        --input "$IMERG_SOURCE" --start "$START" --end "$END" \
        --out "$IMERG" --report "${IMERG%.nc}_qc.json"
fi
for required in "$STATIONS" "$HOLDOUT" "$IMERG" "$CKPT" "$CONFIG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
# Every withheld station must exist in the table this profile assimilates from,
# or scripts/28 aborts late, after the checkpoint load. Catch it here instead.
"$PYTHON_BIN" - "$STATIONS" "$HOLDOUT" <<'PY'
import sys
import pandas as pd

table = pd.read_csv(sys.argv[1], usecols=["station_id"])
present = set(table["station_id"].astype(str))
requested = [line.strip() for line in open(sys.argv[2]) if line.strip()]
missing = sorted(set(requested) - present)
if missing:
    raise SystemExit(
        f"ERROR: {len(missing)} withheld station(s) absent from {sys.argv[1]}, "
        f"e.g. {missing[:5]}"
    )
print(f"[check] all {len(requested)} withheld stations present in {sys.argv[1]}")
PY
if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" ]]; then
    echo "$LABEL already complete; reusing ${PREFIX}.{npz,json}"
    exit 0
fi
if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
    echo "ERROR: partial output at $PREFIX; inspect before retrying" >&2
    exit 1
fi

echo "dense-gauge sweep profile $LABEL (task $TASK_ID)"
echo "  period:   $START through $END; members: $MEMBERS; seed: $SEED"
echo "  group:    $GROUP"
echo "  stations: $STATIONS"
echo "  overrides:${EXTRA_SET[*]:-  (none, frozen production config)}"

"$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --stations "$STATIONS" --imerg "$IMERG" \
    --start "$START" --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" --min-coverage 0.0 \
    --holdout-folds 1 --holdout-fold 0 --holdout-station-ids-file "$HOLDOUT" \
    --group "$GROUP" \
    --set observations.imerg.factor=8 --set observations.imerg.error_corr_cells=0.75 \
    "${EXTRA_SET[@]+"${EXTRA_SET[@]}"}" \
    --seed "$SEED" --out "${PREFIX}.npz" --report "${PREFIX}.json"

echo "Completed profile $LABEL: ${PREFIX}.{npz,json}"
