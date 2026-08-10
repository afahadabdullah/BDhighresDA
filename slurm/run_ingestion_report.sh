#!/usr/bin/env bash
# The whole ingestion evaluation, in one command.
#
# The per-arm SLURM summary jobs run scripts/20 and 21 for ONE configuration
# each: eleven separate reports and no cross-arm comparison. This produces the
# comparison -- the matrix, the significance, and the figures -- from the NPZ
# dumps once the arrays have finished.
#
# Order matters. scripts/42 runs per arm and writes one JSON each; scripts/45
# consolidates those into the single table, so 42 must run first for every arm
# or 45's columns come out blank.
#
# Usage
#   bash slurm/run_ingestion_report.sh                  # after the arrays finish
#   TAG=ing2022 bash slurm/run_ingestion_report.sh
#   SKIP_FIGURES=1 bash slurm/run_ingestion_report.sh   # matrix only, ~1 minute

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

TAG="${TAG:-ing2022}"
REFERENCE="${REFERENCE:-${TAG}_RAW}"
DUMPS="${DUMPS:-data/processed/bmd_imerg_eval_${TAG}_*/*.npz}"
OUT="${OUT:-data/processed/${TAG}_report}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

shopt -s nullglob
found=($DUMPS)
shopt -u nullglob
if [[ ${#found[@]} -eq 0 ]]; then
    echo "ERROR: no dumps match $DUMPS" >&2
    echo "  The arrays may still be running. Check with:" >&2
    echo "    squeue -u \$USER -h -r -t RUNNING,PENDING | wc -l" >&2
    exit 1
fi
configs=$(printf '%s\n' "${found[@]}" | xargs -n1 dirname | sort -u | wc -l)
echo "Found ${#found[@]} dump(s) across $configs configuration(s)"
echo

# ---------------------------------------------------------------- per-arm
# --vs-arm pairs two arms of the SAME dump, so the comparison shares the fold,
# the withheld stations, the days and the seeds. combined-vs-gauges is the
# experiment's question; the other two say what each stream contributes over
# the background it started from.
#
# background runs with no --vs-arm: there is nothing beneath it to pair
# against, and it exists to supply the baseline column. An arm that does not
# beat the background has added nothing, which is worth seeing before any
# argument about which arm is best.
for pair in "background:" "combined:gauges" "gauges:background" "satellite:background"; do
    arm="${pair%%:*}"; vs="${pair##*:}"
    if [[ -n "$vs" ]]; then
        echo "=== scripts/42: $arm versus $vs"
        set -- --arm "$arm" --vs-arm "$vs"
    else
        echo "=== scripts/42: $arm (baseline column)"
        set -- --arm "$arm"
    fi
    "$PYTHON_BIN" -u scripts/42_select_best_config.py \
        --dumps "$DUMPS" "$@" --reference "$REFERENCE" \
        --out-dir "$OUT/selection_$arm"
    echo
done

# -------------------------------------------------------------- the matrix
echo "=== scripts/45: consolidated matrix"
"$PYTHON_BIN" -u scripts/45_ingestion_matrix.py \
    --selections "$OUT/selection_*/config_selection.json" \
    --out-dir "$OUT"
echo

if [[ "$SKIP_FIGURES" == "1" ]]; then
    echo "SKIP_FIGURES=1; stopping after the matrix."
    exit 0
fi

# ------------------------------------------------------------------ figures
echo "=== scripts/40: spatial structure against the product envelope"
"$PYTHON_BIN" -u scripts/40_spatial_structure_screen.py \
    --dumps "${DUMPS%/*}/*fold0*.npz" \
    --arm combined --out-dir "$OUT/structure" || \
    echo "  (scripts/40 failed; the matrix above is unaffected)"
echo

echo "=== scripts/38: aggregation, per-station bias, monthly means"
"$PYTHON_BIN" -u scripts/38_multiyear_gauge_evaluation.py \
    --dumps "data/processed/bmd_imerg_eval_${REFERENCE}/*.npz" \
    --out-dir "$OUT/figures" || \
    echo "  (scripts/38 failed; the matrix above is unaffected)"
echo

# scripts/37 plots ONE configuration, so it needs the winner. Read it out of
# the matrix rather than guessing, and only if an arm actually won.
BEST="$("$PYTHON_BIN" - <<PY
import json, pathlib
p = pathlib.Path("$OUT/ingestion_matrix.json")
rows = json.loads(p.read_text())["rows"] if p.exists() else []
winners = [r for r in rows if r.get("helps")]
print(winners[0]["config"] if winners else "")
PY
)"
if [[ -n "$BEST" ]]; then
    echo "=== scripts/37: per-day maps for the winning configuration ($BEST)"
    "$PYTHON_BIN" -u scripts/37_plot_best_config.py \
        --dumps "data/processed/bmd_imerg_eval_${BEST}/*.npz" \
        --out-prefix "$OUT/best_${BEST}" || \
        echo "  (scripts/37 failed; the matrix above is unaffected)"
else
    echo "=== scripts/37 skipped: no arm significantly beat gauges-only, so"
    echo "    there is no winner to plot. That is the result, not an error."
fi

echo
echo "============================================================"
echo " Report written to $OUT/"
echo "   ingestion_matrix.md    the table to read first"
echo "   ingestion_matrix.png   three panels: skill, significance, structure"
echo "   ingestion_matrix.csv   same numbers, for the paper"
echo "   selection_*/           per-arm detail from scripts/42"
echo "   structure/ figures/    spectra, aggregation, per-station bias"
echo "============================================================"
