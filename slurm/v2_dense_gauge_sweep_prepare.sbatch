#!/usr/bin/env bash
#SBATCH --job-name=bdhires-dense-prep
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Build the station table for the dense-network method sweep, and measure the
# gauge error budget the sweep is about.
#
# The station table and the holdout are produced with the SAME period, seed and
# constraints as the 2022 season of the frozen BMD+BWDB production archive
# (scripts/82, seed V2_BWDB_WINNER_HOLDOUT_SEED + 1).  That is deliberate: it
# reproduces the production station set and fold exactly, so a method compared
# here is compared on the network the production run actually used, not on a
# five-day re-derivation of it.  The assimilation window is then narrowed to the
# sweep's five days.

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

ROOT="${V2_DENSE_ROOT:-data/processed/v2_dense_gauge_sweep}"
BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
STATS="${V2_DENSE_STATS:-data/processed/stats.json}"
SEASON_START="${V2_DENSE_SEASON_START:-2022-05-01}"
SEASON_END="${V2_DENSE_SEASON_END:-2022-09-30}"
HOLDOUT_SEED="${V2_DENSE_HOLDOUT_SEED:-202122}"
CELL_DEG="${V2_DENSE_CELL_DEG:-0.25}"
SIGMA_OBS="${V2_DENSE_SIGMA_OBS:-0.10}"
START="${V2_DENSE_START:-2022-05-01}"
END="${V2_DENSE_END:-2022-05-05}"
IMERG_SOURCE="${V2_DENSE_IMERG:-data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc}"
IMERG_SUBSET="$ROOT/imerg_s04_${START//-/}_${END//-/}.nc"

for required in "$BMD_STATIONS" "$BWDB_XLSX" "$STATS" "$IMERG_SOURCE"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD data directory: $BMD_DATA_DIR"; exit 1; }

STATION_ROOT="$ROOT/stations"
mkdir -p "$STATION_ROOT/holdout"

if [[ ! -s "$IMERG_SUBSET" ]]; then
    "$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
        --input "$IMERG_SOURCE" --start "$START" --end "$END" \
        --out "$IMERG_SUBSET" --report "${IMERG_SUBSET%.nc}_qc.json"
else
    echo "Reusing $IMERG_SUBSET"
fi

if [[ -s "$STATION_ROOT/combined_daily.csv" && -s "$STATION_ROOT/holdout/fold0.txt" ]]; then
    echo "Reusing $STATION_ROOT/combined_daily.csv"
else
    "$PYTHON_BIN" -u scripts/82_prepare_bmd_bwdb_may2022.py \
        --bmd-data-dir "$BMD_DATA_DIR" --bmd-stations "$BMD_STATIONS" \
        --bwdb-xlsx "$BWDB_XLSX" \
        --start "$SEASON_START" --end "$SEASON_END" \
        --out "$STATION_ROOT/combined_daily.csv" \
        --summary "$STATION_ROOT/station_summary.csv" \
        --report "$STATION_ROOT/preparation_manifest.json" \
        --holdout-dir "$STATION_ROOT/holdout" \
        --holdout-folds 1 --holdout-fraction 0.20 \
        --holdout-neighbor-km 15 --coincident-group-km 5 \
        --seed "$HOLDOUT_SEED"
fi

# scripts/82 writes every station it parsed; the production run then dropped
# the ineligible ones through load_stations' 0.5 coverage filter over the whole
# May-September season.  A five-day window cannot reproduce that filter -- a
# station present all season but quiet in the first five days would be dropped,
# and the season-derived holdout list would then fail to resolve.  So the
# eligibility decision is made once here, on the season, and the sweep runs the
# filtered table with --min-coverage 0.  The assimilated set is therefore
# exactly the production set, and a gauge that simply did not report on one of
# the five days contributes nothing on that day rather than vanishing.
"$PYTHON_BIN" - "$STATION_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
daily = pd.read_csv(root / "combined_daily.csv", parse_dates=["date"])
summary = pd.read_csv(root / "station_summary.csv")
eligible = set(summary.loc[summary["eligible_for_analysis"], "station_id"].astype(str))
daily["station_id"] = daily["station_id"].astype(str)
filtered = daily[daily["station_id"].isin(eligible)]
missing = eligible - set(filtered["station_id"])
if missing:
    raise SystemExit(f"eligible stations absent from the daily table: {sorted(missing)[:5]}")
filtered.to_csv(root / "combined_eligible.csv", index=False)
print(
    f"[prepare] eligible station table: {filtered['station_id'].nunique()} of "
    f"{daily['station_id'].nunique()} stations -> combined_eligible.csv"
)
PY

# The measured error budget for the network as it is actually assimilated: one
# point gauge per station, one scalar R for both sources.
"$PYTHON_BIN" -u scripts/87_superob_dense_gauges.py --report-only \
    --stations "$STATION_ROOT/combined_eligible.csv" \
    --holdout-ids "$STATION_ROOT/holdout/fold0.txt" \
    --stats "$STATS" --sigma-obs "$SIGMA_OBS" --cell-deg "$CELL_DEG" \
    --report "$STATION_ROOT/error_budget.json"

# The aggregated station table for the super-observation profile. The floor
# comes from the source fold's own retained-neighbour minimum, so aggregation
# cannot make the withheld scoring easier than it is for every other profile.
FLOOR="$("$PYTHON_BIN" -c "
import json
manifest = json.load(open('$STATION_ROOT/preparation_manifest.json'))
print(manifest['analysis_selection']['minimum_retained_neighbour_km'])
")"
"$PYTHON_BIN" -u scripts/87_superob_dense_gauges.py \
    --stations "$STATION_ROOT/combined_eligible.csv" \
    --holdout-ids "$STATION_ROOT/holdout/fold0.txt" \
    --stats "$STATS" --sigma-obs "$SIGMA_OBS" --cell-deg "$CELL_DEG" \
    --protect-withheld-km "$FLOOR" --fail-under-km "$FLOOR" \
    --out "$STATION_ROOT/superob_${CELL_DEG}.csv" \
    --report "$STATION_ROOT/superob_${CELL_DEG}.json"

echo "Prepared dense-network sweep inputs under $STATION_ROOT"
