#!/usr/bin/env bash
# Every paper figure up to and including the OSSE, in one command.
#
# Outline: docs/PAPER_OUTLINE.md
#
# Figures 1-3 (method and design) need only the station catalogue and the
# existing schematic, so they build in seconds without any model run.
# Figures 4-7 and Table 1 need OSSE output; if it is absent each figure says
# so and is skipped rather than failing the run, so the method figures are
# still produced.
#
# THE REAL-DATA FIGURES (8-10) ARE NOT BUILT HERE. They are deferred by
# request; scripts/45 and slurm/run_ingestion_report.sh already produce them.
#
# WINDOW: 2021-01-01..2024-12-31, not 2020 onward. The prior's split is
# train [1981,2018], val [2019,2020], test [2021,2025], so including 2020
# would score on years the checkpoint selection saw. 2021-2024 is strictly
# out of sample and happens to match the prepared IMERG archive exactly.
#
# Usage
#   bash slurm/make_paper_figures.sh
#   OUT=docs/paper_figures OSSE_ROOT=data/processed/osse_paper \
#       bash slurm/make_paper_figures.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

START="${START:-2021-01-01}"
END="${END:-2024-12-31}"
OUT="${OUT:-docs/paper_figures}"
STATIONS_WAS_SET="${STATIONS+x}"
STATIONS="${STATIONS:-data/stations/bmd_daily.csv}"
BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATION_CATALOG="${BMD_STATION_CATALOG:-$BMD_DATA_DIR/Stations.csv}"
OSSE_ROOT="${OSSE_ROOT:-data/processed/osse_paper}"
PRIMARY="${PRIMARY:-simultaneous_realistic_40}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT" "$OUT/data"

# The repository stores recent BMD observations as one CSV per station.  The
# plotting code consumes the canonical long-form table.  Build that table on
# demand instead of assuming that an earlier DA run happened to create it.
if [[ ! -s "$STATIONS" ]]; then
    if [[ -n "$STATIONS_WAS_SET" ]]; then
        echo "ERROR: explicitly requested station file does not exist: $STATIONS" >&2
        exit 1
    fi
    if [[ ! -d "$BMD_DATA_DIR" || ! -s "$BMD_STATION_CATALOG" ]]; then
        echo "ERROR: station file $STATIONS is absent and the source archive is incomplete." >&2
        echo "Expected per-station CSVs in $BMD_DATA_DIR and catalogue $BMD_STATION_CATALOG" >&2
        echo "Alternatively set STATIONS=/path/to/canonical_bmd_daily.csv" >&2
        exit 1
    fi
    START_KEY="${START//-/}"
    END_KEY="${END//-/}"
    STATIONS="$OUT/data/bmd_daily_${START_KEY}_${END_KEY}.csv"
    STATION_SUMMARY="$OUT/data/bmd_stations_${START_KEY}_${END_KEY}.csv"
    STATION_QC="$OUT/data/bmd_qc_${START_KEY}_${END_KEY}.json"
    if [[ -s "$STATIONS" && -s "$STATION_SUMMARY" && -s "$STATION_QC" ]]; then
        echo "Reusing canonical BMD table: $STATIONS"
    else
        echo "Canonical BMD table is absent; converting $BMD_DATA_DIR"
        "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
            --data-dir "$BMD_DATA_DIR" \
            --stations "$BMD_STATION_CATALOG" \
            --start "$START" \
            --end "$END" \
            --out "$STATIONS" \
            --summary "$STATION_SUMMARY" \
            --report "$STATION_QC"
    fi
fi

echo "============================================================"
echo " Paper figures 1-7 + Table 1"
echo "   window   $START .. $END  (strict test split)"
echo "   stations $STATIONS"
echo "   osse     $OSSE_ROOT"
echo "   out      $OUT/  (data in $OUT/data/)"
echo "============================================================"
echo

echo "--- Figures 1-3: pipeline, observation error, evaluation design"
"$PYTHON_BIN" -u scripts/46_paper_figures.py \
    --stations "$STATIONS" --start "$START" --end "$END" \
    --out-dir "$OUT"
echo

echo "--- Figures 4-7 + Table 1: OSSE"
if [[ -d "$OSSE_ROOT" ]]; then
    "$PYTHON_BIN" -u scripts/47_osse_paper_figures.py \
        --root "$OSSE_ROOT" --primary "$PRIMARY" --out-dir "$OUT" || {
        echo "  scripts/47 failed; figures 1-3 above are unaffected." >&2
    }
else
    echo "  $OSSE_ROOT does not exist yet. Run the OSSE first:"
    echo "    OSSE_PAPER_START=$START OSSE_PAPER_END=$END \\"
    echo "        bash slurm/submit_osse_paper.sh"
    echo "  then re-run this script."
fi

echo
echo "============================================================"
ls -1 "$OUT"/fig*.pdf "$OUT"/fig*.png "$OUT"/table*.tex 2>/dev/null | sed 's/^/  /'
echo
echo " Data behind every panel: $OUT/data/*.csv"
echo " Provenance (script, inputs, git commit): $OUT/data/*_manifest.json"
echo "============================================================"
