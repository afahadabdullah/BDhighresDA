#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-checker
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_huber3_2021_2024}"
REPORT_OUT="${ROOT}/data_integrity_report.json"

echo "Running comprehensive BMD+BWDB 2021-2024 archive integrity check..."
echo "Archive root: $ROOT"
echo "Report output: $REPORT_OUT"

"$PYTHON_BIN" -u scripts/86_check_bmd_bwdb_2021_2024_archive.py \
    --root "$ROOT" \
    --out "$REPORT_OUT" \
    --check-all-zarr-members

echo "Data integrity checker completed successfully!"
