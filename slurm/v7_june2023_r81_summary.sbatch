#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-jun23-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"

ROOT="${V7_JUNE_ROOT:-data/processed/v7_june2023_r81_latest_latest}"
CPC_ROOT="${V7_JUNE_CPC_ROOT:-data/processed/v2_confirmatory_2021_2024}"
V7_DUMPS=()
CPC_DUMPS=()
for fold in 0 1 2 3 4; do
    v7="$ROOT/cv/june2023/fold${fold}/station_ensembles.npz"
    cpc="$CPC_ROOT/cv/2023_may_sep/fold${fold}.npz"
    [[ -s "$v7" ]] || { echo "ERROR: missing $v7" >&2; exit 1; }
    [[ -s "$cpc" ]] || { echo "ERROR: missing $cpc" >&2; exit 1; }
    V7_DUMPS+=("$v7")
    CPC_DUMPS+=("$cpc")
done
[[ -d "$ROOT/gridded/june2023.zarr" ]] || {
    echo "ERROR: missing $ROOT/gridded/june2023.zarr" >&2
    exit 1
}

"$PYTHON_BIN" -u scripts/79_compare_v7_cpcv2_folds.py \
    --v7-dumps "${V7_DUMPS[@]}" \
    --cpcv2-dumps "${CPC_DUMPS[@]}" \
    --bootstrap-resamples "${V7_JUNE_BOOTSTRAPS:-10000}" \
    --seed "${V7_JUNE_SEED:-201805}" \
    --out-json "$ROOT/comparison/june2023_v7_r81_vs_cpcv2.json" \
    --out-markdown "$ROOT/comparison/june2023_v7_r81_vs_cpcv2.md"

echo "comparison complete: $ROOT/comparison/june2023_v7_r81_vs_cpcv2.md"
