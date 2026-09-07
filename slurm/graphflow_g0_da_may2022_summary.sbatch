#!/usr/bin/env bash
#SBATCH --job-name=graphflow-g0-da-summary
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

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

ROOT="${GRAPHFLOW_DA_ROOT:-runs/graphflow_g0_multimesh/da_may2022}"
CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
SUMMARY="$ROOT/summary"

CPC_DUMPS=() CPC_REPORTS=() GRAPH_DUMPS=() GRAPH_REPORTS=()
for fold in 0 1 2 3 4; do
    CPC_DUMPS+=("$CPC_ROOT/fold${fold}.npz")
    CPC_REPORTS+=("$CPC_ROOT/fold${fold}.json")
    GRAPH_DUMPS+=("$ROOT/graphflow_g0_multimesh/fold${fold}.npz")
    GRAPH_REPORTS+=("$ROOT/graphflow_g0_multimesh/fold${fold}.json")
done

"$PYTHON_BIN" -u scripts/summarize_graphflow_da.py \
    --cpc-dumps "${CPC_DUMPS[@]}" --cpc-reports "${CPC_REPORTS[@]}" \
    --graph-dumps "${GRAPH_DUMPS[@]}" --graph-reports "${GRAPH_REPORTS[@]}" \
    --out-json "$SUMMARY/comparison.json" \
    --out-markdown "$SUMMARY/comparison.md" \
    --out-plot "$SUMMARY/comparison.png"
