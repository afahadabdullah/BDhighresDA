#!/usr/bin/env bash
#SBATCH --job-name=cpc-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

START="${CPC_START:-1981}"
END="${CPC_END:-2025}"
OUT_DIR="${CPC_OUT:-data/raw/cpc}"
PYTHON_BIN="${CPC_PYTHON:-python3}"

[[ -f scripts/02b_download_cpc.py ]] || {
    echo "ERROR: Submit from the BDhighresDA repository root."
    exit 1
}

mkdir -p "$OUT_DIR"
echo "Downloading complete CPC record $START-$END to $OUT_DIR"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/02b_download_cpc.py \
    --start "$START" \
    --end "$END" \
    --out "$OUT_DIR" \
    --require-complete
