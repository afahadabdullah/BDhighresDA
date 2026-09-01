#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-may-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_may2022}"
DUMPS=() REPORTS=()
for fold in 0 1 2; do
    DUMPS+=("$ROOT/cv/fold${fold}.npz")
    REPORTS+=("$ROOT/cv/fold${fold}.json")
done
"$PYTHON_BIN" -u scripts/83_summarize_bmd_bwdb_may2022.py \
    --dumps "${DUMPS[@]}" --reports "${REPORTS[@]}" \
    --manifest "$ROOT/stations/preparation_manifest.json" \
    --station-summary "$ROOT/stations/station_summary.csv" \
    --out-dir "$ROOT/summary"
echo "Summary: $ROOT/summary/may2022_bmd_bwdb_scores.{md,json}"
